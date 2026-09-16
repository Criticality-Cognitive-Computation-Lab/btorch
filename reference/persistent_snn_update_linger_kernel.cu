#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace cg = cooperative_groups;

namespace {

#ifndef BTORCH_BLOCK_EDGE_BUDGET
#define BTORCH_BLOCK_EDGE_BUDGET 0
#endif

#ifndef BTORCH_LONG_SEGMENT_SIZE
#define BTORCH_LONG_SEGMENT_SIZE 1024
#endif

#ifndef BTORCH_TILE_REDUCE_MODE
#define BTORCH_TILE_REDUCE_MODE 0
#endif

#ifndef BTORCH_BLOCK_HASH_AGGREGATION
#define BTORCH_BLOCK_HASH_AGGREGATION 512
#endif

#ifndef BTORCH_BLOCK_HASH_CAPACITY
#define BTORCH_BLOCK_HASH_CAPACITY 512
#endif

#ifndef BTORCH_BLOCK_HASH_MAX_PROBE
#define BTORCH_BLOCK_HASH_MAX_PROBE 4
#endif

#ifndef BTORCH_BLOCK_HASH_MIN_EDGES
#define BTORCH_BLOCK_HASH_MIN_EDGES 256
#endif

#ifndef BTORCH_BLOCK_HASH_USED_SLOTS
#define BTORCH_BLOCK_HASH_USED_SLOTS 1
#endif

constexpr int kFanoutThreshold = 256;
constexpr int kLaneRowThreshold = 4;
constexpr int kBlockEdgeBudget = BTORCH_BLOCK_EDGE_BUDGET;
constexpr int kSegmentSize = BTORCH_LONG_SEGMENT_SIZE;
constexpr int kTileReduceMode = BTORCH_TILE_REDUCE_MODE;
constexpr int kHashAggregation = BTORCH_BLOCK_HASH_AGGREGATION;
constexpr int kHashSize = BTORCH_BLOCK_HASH_CAPACITY;
constexpr int kHashMaxProbe = BTORCH_BLOCK_HASH_MAX_PROBE;
constexpr int kHashMinEdges = BTORCH_BLOCK_HASH_MIN_EDGES;
constexpr bool kHashUsedSlots = BTORCH_BLOCK_HASH_USED_SLOTS != 0;
constexpr int kLegacyHashMaxEdges = 96;
constexpr int kWarpsPerBlock = 8;
constexpr int kMaxUpdateNeuronsPerBlock = 256;
constexpr int kEmptyHashKey = -1;
constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kBlockIndexBits = 24;
constexpr unsigned kBlockIndexMask = (1u << kBlockIndexBits) - 1u;
#ifdef ENABLE_BLOCK_HASH
constexpr int kBlockTaskSpan =
    kHashAggregation > kBlockEdgeBudget
    ? kHashAggregation
    : kBlockEdgeBudget;
#else
constexpr int kBlockTaskSpan = kBlockEdgeBudget;
#endif

static_assert(
    kBlockEdgeBudget == 0
        || kBlockEdgeBudget == 128
        || kBlockEdgeBudget == 256
        || kBlockEdgeBudget == 512
        || kBlockEdgeBudget == 1024);
static_assert(
    kSegmentSize == 128
        || kSegmentSize == 256
        || kSegmentSize == 512
        || kSegmentSize == 1024
        || kSegmentSize == 2048);
static_assert(kTileReduceMode >= 0 && kTileReduceMode <= 2);
static_assert(
    kHashAggregation == 128
        || kHashAggregation == 256
        || kHashAggregation == 512);
static_assert(kHashSize == 128 || kHashSize == 256 || kHashSize == 512);
static_assert(
    kHashMaxProbe == 4 || kHashMaxProbe == 8 || kHashMaxProbe == 16);
static_assert(
    kHashMinEdges == 0
        || kHashMinEdges == 64
        || kHashMinEdges == 128
        || kHashMinEdges == 192
        || kHashMinEdges == 256);
static_assert(
    BTORCH_BLOCK_HASH_USED_SLOTS == 0
        || BTORCH_BLOCK_HASH_USED_SLOTS == 1);

#ifdef ENABLE_BLOCK_STATS
constexpr int kBlockStatsColumns = 23;
constexpr int kV4InputEdgesColumn = 13;
constexpr int kV4GlobalAtomicsColumn = 14;
constexpr int kV4ReduceTasksColumn = 15;
constexpr int kV4ReduceEdgesColumn = 16;
constexpr int kHashTasksColumn = 17;
constexpr int kHashWindowsColumn = 18;
constexpr int kHashInputEdgesColumn = 19;
constexpr int kHashFlushAtomicsColumn = 20;
constexpr int kHashFallbackAtomicsColumn = 21;
constexpr int kHashProbeAttemptsColumn = 22;

// One diagnostic thread reconstructs one potential 32-neuron task after the
// persistent launch. This deliberately lives outside the timed/production
// path; the normal build contains neither this kernel nor its output buffer.
__global__ void collect_block_stats_kernel(
    const float* __restrict__ dense_spikes,
    const int* __restrict__ graph_indptr,
    int* __restrict__ stats,
    int t_steps,
    int batch_size,
    int n_neuron,
    int blocks_per_batch) {
    const int record = blockIdx.x * blockDim.x + threadIdx.x;
    const int record_count = t_steps * batch_size * blocks_per_batch;
    if (record >= record_count) {
        return;
    }
    const int block = record % blocks_per_batch;
    const int bucket = record / blocks_per_batch;
    const int block_start = block * 32;
    const int block_end = min(block_start + 32, n_neuron);
    const float* bucket_spikes = dense_spikes + bucket * n_neuron;

    unsigned active_mask = 0;
    int active_rows = 0;
    int active_edges = 0;
    int active_runs = 0;
    int max_row_degree = 0;
    int first_active = -1;
    int last_active = -1;
    int long_segment_tasks = 0;
    bool previous_active = false;
    bool all_direct_lane = true;
    for (int neuron = block_start; neuron < block_end; ++neuron) {
        if (bucket_spikes[neuron] == 0.0f) {
            previous_active = false;
            continue;
        }
        const int degree = graph_indptr[neuron + 1] - graph_indptr[neuron];
        if (degree >= kFanoutThreshold) {
            long_segment_tasks +=
                (degree + kSegmentSize - 1) / kSegmentSize;
            previous_active = false;
            continue;
        }
        const int lane = neuron - block_start;
        active_mask |= 1u << lane;
        ++active_rows;
        active_edges += degree;
        max_row_degree = max(max_row_degree, degree);
        all_direct_lane = all_direct_lane && degree <= kLaneRowThreshold;
        if (!previous_active) {
            ++active_runs;
        }
        previous_active = true;
        first_active = first_active < 0 ? neuron : first_active;
        last_active = neuron;
    }

    int* row = stats + record * kBlockStatsColumns;
    row[0] = active_rows;
    row[1] = active_edges;
    row[2] = active_runs;
    row[3] = first_active < 0
        ? 0
        : graph_indptr[last_active + 1] - graph_indptr[first_active];
    row[4] = max_row_degree;
    row[5] = active_edges;  // current spike-block path: one atomic per edge
    row[6] = active_rows > 0 && all_direct_lane ? 1 : 0;
    row[7] = active_rows > active_runs ? 1 : 0;
    row[8] = long_segment_tasks;
    row[9] = bucket / batch_size;
    row[10] = bucket % batch_size;
    row[11] = block_start;
    row[12] = static_cast<int>(active_mask);
}
#endif

__device__ __forceinline__ void process_edge_range(
    const int* graph_indices,
    const float* graph_weight,
    float* psc,
    int start,
    int end,
    int lane) {
    for (int edge_base = start; edge_base < end; edge_base += 32) {
        const int edge = edge_base + lane;
        const bool valid = edge < end;
        if (valid) {
            const int post = graph_indices[edge];
            atomicAdd(psc + post, graph_weight[edge]);
        }
    }
}

__device__ __forceinline__ int map_packed_edge(
    const int* packed_prefix,
    const int* packed_starts,
    int logical_edge) {
    int low = 0;
    int high = 31;
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int middle = (low + high) >> 1;
        if (packed_prefix[middle] > logical_edge) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    const int previous_end = low == 0 ? 0 : packed_prefix[low - 1];
    return packed_starts[low] + logical_edge - previous_end;
}

template <bool ReturnDense>
__global__ void persistent_snn_spike_block_kernel(
    const int* __restrict__ event_offsets,
    const int* __restrict__ event_indices,
    const float* __restrict__ event_values,
    bool has_event_values,
    const int* __restrict__ graph_indptr,
    const int* __restrict__ graph_indices,
    const float* __restrict__ graph_weight,
    float* __restrict__ v,
    float* __restrict__ psc,
    float* __restrict__ dense_spikes,
    float* __restrict__ input_current,
    int* __restrict__ task_queue_batch,
    int* __restrict__ task_queue_start,
    int* __restrict__ task_queue_end_or_mask,
    int* __restrict__ task_counts,
    int* __restrict__ work_counters,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
    int* __restrict__ block_stats,
    bool return_events,
    int t_steps,
    int batch_size,
    int n_neuron,
    int queue_capacity,
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m) {
    cg::grid_group grid = cg::this_grid();
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int lane = global_tid & 31;
    const int warp_in_block = threadIdx.x >> 5;
    // Every CTA retains its existing propagation role. During UPDATE, the
    // cooperative grid also provides a fixed, contiguous neuron ownership.
    const int update_block_count = gridDim.x;
    const bool is_update_block = blockIdx.x < update_block_count;
    const int neurons_per_update_block =
        (n_neuron + update_block_count - 1) / update_block_count;
    const int neuron_begin = blockIdx.x * neurons_per_update_block;
    const int neuron_end = min(
        neuron_begin + neurons_per_update_block, n_neuron);
    // A single shared buffer can retain the complete voltage state only for
    // the production B=1 path. Other shapes keep the same static ownership,
    // but safely use global voltage storage.
    const bool resident_v =
        batch_size == 1
        && neurons_per_update_block <= kMaxUpdateNeuronsPerBlock;
    __shared__ float shared_v[kMaxUpdateNeuronsPerBlock];
    __shared__ int packed_prefix[kWarpsPerBlock][32];
    __shared__ int packed_starts[kWarpsPerBlock][32];
#ifdef ENABLE_BLOCK_HASH
    __shared__ int hash_keys[kWarpsPerBlock][kHashSize];
    __shared__ float hash_values[kWarpsPerBlock][kHashSize];
    __shared__ unsigned short
        hash_used_slots[kWarpsPerBlock][kHashSize];
#else
    __shared__ int hash_keys[1][1];
    __shared__ float hash_values[1][1];
    __shared__ unsigned short hash_used_slots[1][1];
#endif
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;

    if (is_update_block && resident_v) {
        for (int n = neuron_begin + threadIdx.x;
             n < neuron_end;
             n += blockDim.x) {
            shared_v[n - neuron_begin] = v[n];
        }
    }
    __syncthreads();

#ifdef ENABLE_BLOCK_HASH
    if constexpr (kHashUsedSlots) {
        for (int hash_slot = lane;
             hash_slot < kHashSize;
             hash_slot += 32) {
            hash_keys[warp_in_block][hash_slot] = kEmptyHashKey;
        }
        __syncwarp();
    }
#endif

    for (int t = 0; t < t_steps; ++t) {
        if (global_tid == 0) {
            task_counts[0] = 0;
            task_counts[1] = 0;
            work_counters[0] = 0;
            work_counters[1] = 0;
        }
        for (int b = 0; b < batch_size; ++b) {
            const int bucket = t * batch_size + b;
            const int start = event_offsets[bucket];
            const int end = event_offsets[bucket + 1];
            for (int event = start + global_tid; event < end; event += stride) {
                const int pre = event_indices[event];
                if (pre >= 0 && pre < n_neuron) {
                    const float value =
                        has_event_values ? event_values[event] : 1.0f;
                    atomicAdd(input_current + b * n_neuron + pre, value);
                }
            }
        }
        grid.sync();

        for (int b = 0; b < batch_size; ++b) {
            const int batch_base = b * n_neuron;
            if (is_update_block) {
                for (int n = neuron_begin + threadIdx.x;
                     n < neuron_end;
                     n += blockDim.x) {
                    const int cell = batch_base + n;
                    const float current = psc[cell] + input_current[cell];
                    input_current[cell] = 0.0f;
                    const int local_n = n - neuron_begin;
                    const float voltage = resident_v
                        ? shared_v[local_n]
                        : v[cell];
                    const float v_pre = voltage +
                        dt * (-(voltage - v_reset) / tau_mem + current / c_m);
                    const bool fired = v_pre >= v_threshold;
                    const float spike = fired ? 1.0f : 0.0f;
                    const float voltage_next = v_pre - reset_delta * spike;
                    if (resident_v) {
                        shared_v[local_n] = voltage_next;
                    } else {
                        v[cell] = voltage_next;
                    }
                    if constexpr (ReturnDense) {
                        dense_spikes[
                            (t * batch_size + b) * n_neuron + n] = spike;
                    }
                    psc[cell] *= decay;

                    const int edge_start = graph_indptr[n];
                    const int edge_end = graph_indptr[n + 1];
                    const int degree = edge_end - edge_start;
                    const bool block_spike =
                        fired && degree > 0 && degree < kFanoutThreshold;
                    const unsigned warp_active = __activemask();
                    const unsigned spike_mask =
                        __ballot_sync(warp_active, block_spike);
                    if constexpr (kBlockEdgeBudget > 0) {
                        int inclusive_edges = block_spike ? degree : 0;
    #pragma unroll
                        for (int offset = 1; offset < 32; offset <<= 1) {
                            const int value = __shfl_up_sync(
                                warp_active, inclusive_edges, offset);
                            if (lane >= offset) {
                                inclusive_edges += value;
                            }
                        }
                        const int final_lane = 31 - __clz(warp_active);
                        const int active_edges = __shfl_sync(
                            warp_active, inclusive_edges, final_lane);
                        if (lane == 0 && active_edges > 0) {
                            const int logical_task_count =
                                (active_edges + kBlockTaskSpan - 1)
                                / kBlockTaskSpan;
                            const int first_task = atomicAdd(
                                task_counts + 1, logical_task_count);
                            // Static ownership is deliberately not 32-neuron
                            // aligned, so retain the exact first neuron of the
                            // producing warp in the packed descriptor.
                            const int block_start = n;
                            for (int segment = 0;
                                 segment < logical_task_count;
                                 ++segment) {
                                const int task = first_task + segment;
                                const int slot = queue_capacity - 1 - task;
                                const unsigned descriptor =
                                    static_cast<unsigned>(block_start)
                                    | (static_cast<unsigned>(segment)
                                       << kBlockIndexBits);
                                task_queue_batch[slot] = b;
                                task_queue_start[slot] =
                                    static_cast<int>(descriptor);
                                task_queue_end_or_mask[slot] =
                                    static_cast<int>(spike_mask);
                            }
                        }
                    } else if (lane == 0 && spike_mask != 0) {
                        const int task = atomicAdd(task_counts + 1, 1);
                        const int slot = queue_capacity - 1 - task;
                        task_queue_batch[slot] = b;
                        task_queue_start[slot] = n;
                        task_queue_end_or_mask[slot] =
                            static_cast<int>(spike_mask);
                    }

                    if (fired && degree >= kFanoutThreshold) {
                        const int segment_count =
                            (degree + kSegmentSize - 1) / kSegmentSize;
                        const int first_task =
                            atomicAdd(task_counts, segment_count);
                        for (int segment = 0;
                             segment < segment_count;
                             ++segment) {
                            const int task = first_task + segment;
                            const int segment_start =
                                edge_start + segment * kSegmentSize;
                            task_queue_batch[task] = b;
                            task_queue_start[task] = segment_start;
                            task_queue_end_or_mask[task] =
                                min(segment_start + kSegmentSize, edge_end);
                        }
                    }

                    if (fired && return_events) {
                        const int bucket = t * batch_size + b;
                        const int rank = atomicAdd(event_counts + bucket, 1);
                        if (rank < n_neuron) {
                            event_indices_full[bucket * n_neuron + rank] = n;
                        }
                    }
                }
            }
        }
        grid.sync();

        const int segment_count = task_counts[0];
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counters, 1);
            }
            task = __shfl_sync(kFullWarpMask, task, 0);
            if (task >= segment_count) {
                break;
            }
            const int b = task_queue_batch[task];
            process_edge_range(
                graph_indices,
                graph_weight,
                psc + b * n_neuron,
                task_queue_start[task],
                task_queue_end_or_mask[task],
                lane);
        }

        const int block_count = task_counts[1];
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counters + 1, 1);
            }
            task = __shfl_sync(kFullWarpMask, task, 0);
            if (task >= block_count) {
                break;
            }
            const int slot = queue_capacity - 1 - task;
            const int b = task_queue_batch[slot];
            const unsigned descriptor =
                static_cast<unsigned>(task_queue_start[slot]);
            const int block_start = kBlockEdgeBudget > 0
                ? static_cast<int>(descriptor & kBlockIndexMask)
                : static_cast<int>(descriptor);
            const int logical_segment = kBlockEdgeBudget > 0
                ? static_cast<int>(descriptor >> kBlockIndexBits)
                : 0;
            const unsigned spike_mask =
                static_cast<unsigned>(task_queue_end_or_mask[slot]);
            const bool lane_fired =
                (spike_mask & (1u << lane)) != 0;
            const int lane_pre = block_start + lane;
            const int lane_edge_start =
                lane_fired ? graph_indptr[lane_pre] : 0;
            const int lane_edge_end =
                lane_fired ? graph_indptr[lane_pre + 1] : 0;
            const int lane_degree = lane_edge_end - lane_edge_start;
            if constexpr (kBlockEdgeBudget > 0) {
                int inclusive_end = lane_degree;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const int value = __shfl_up_sync(
                        kFullWarpMask, inclusive_end, offset);
                    if (lane >= offset) {
                        inclusive_end += value;
                    }
                }
                packed_prefix[warp_in_block][lane] = inclusive_end;
                packed_starts[warp_in_block][lane] = lane_edge_start;
                __syncwarp();

                const int total_edges = packed_prefix[warp_in_block][31];
                const int logical_begin =
                    logical_segment * kBlockTaskSpan;
                const int logical_end = min(
                    total_edges, logical_begin + kBlockTaskSpan);
                const int task_edges = logical_end - logical_begin;
#ifdef ENABLE_BLOCK_HASH
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0 && block_stats != nullptr) {
                    atomicAdd(
                        block_stats + kV4InputEdgesColumn,
                        task_edges);
                }
#endif
                bool task_used_hash = false;
                for (int window_begin = logical_begin;
                     window_begin < logical_end;
                     window_begin += kHashAggregation) {
                    const int window_end = min(
                        logical_end, window_begin + kHashAggregation);
                    const int window_edges = window_end - window_begin;
                    const bool use_hash = window_edges >= kHashMinEdges;
                    if (!use_hash) {
                        for (int edge_base = window_begin;
                             edge_base < window_end;
                             edge_base += 32) {
                            const int logical_edge = edge_base + lane;
                            if (logical_edge < window_end) {
                                const int edge = map_packed_edge(
                                    packed_prefix[warp_in_block],
                                    packed_starts[warp_in_block],
                                    logical_edge);
                                atomicAdd(
                                    psc + b * n_neuron
                                        + graph_indices[edge],
                                    graph_weight[edge]);
                            }
                        }
#ifdef ENABLE_BLOCK_STATS
                        if (lane == 0 && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kV4GlobalAtomicsColumn,
                                window_edges);
                        }
#endif
                        continue;
                    }

                    task_used_hash = true;
                    if constexpr (!kHashUsedSlots) {
                        for (int hash_slot = lane;
                             hash_slot < kHashSize;
                             hash_slot += 32) {
                            hash_keys[warp_in_block][hash_slot] =
                                kEmptyHashKey;
                            hash_values[warp_in_block][hash_slot] = 0.0f;
                        }
                        __syncwarp();
                    }
                    int used_count = 0;

#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kHashWindowsColumn, 1);
                        atomicAdd(
                            block_stats + kHashInputEdgesColumn,
                            window_edges);
                    }
#endif
                    for (int edge_base = window_begin;
                         edge_base < window_end;
                         edge_base += 32) {
                        const int logical_edge = edge_base + lane;
                        const bool valid = logical_edge < window_end;
                        int post = 0;
                        float weight = 0.0f;
                        if (valid) {
                            const int edge = map_packed_edge(
                                packed_prefix[warp_in_block],
                                packed_starts[warp_in_block],
                                logical_edge);
                            post = graph_indices[edge];
                            weight = graph_weight[edge];
                        }

                        int destination_slot = -1;
                        int probe_attempts = 0;
                        bool claimed_new = false;
                        if (valid) {
                            int hash_slot = static_cast<int>(
                                (static_cast<unsigned>(post)
                                 * 2654435761u)
                                & (kHashSize - 1));
#pragma unroll
                            for (int probe = 0;
                                 probe < kHashMaxProbe;
                                 ++probe) {
                                ++probe_attempts;
                                const int old = atomicCAS(
                                    &hash_keys
                                        [warp_in_block][hash_slot],
                                    kEmptyHashKey,
                                    post);
                                if (old == kEmptyHashKey
                                    || old == post) {
                                    destination_slot = hash_slot;
                                    claimed_new =
                                        old == kEmptyHashKey;
                                    if constexpr (kHashUsedSlots) {
                                        if (claimed_new) {
                                            hash_values
                                                [warp_in_block][hash_slot] =
                                                0.0f;
                                        }
                                    }
                                    break;
                                }
                                hash_slot =
                                    (hash_slot + 1) & (kHashSize - 1);
                            }
                        }

                        if constexpr (kHashUsedSlots) {
                            const unsigned new_key_mask =
                                __ballot_sync(
                                    kFullWarpMask, claimed_new);
                            if (claimed_new) {
                                const unsigned lower_lanes =
                                    lane == 0
                                    ? 0u
                                    : (1u << lane) - 1u;
                                const int used_rank = __popc(
                                    new_key_mask & lower_lanes);
                                hash_used_slots
                                    [warp_in_block]
                                    [used_count + used_rank] =
                                    static_cast<unsigned short>(
                                        destination_slot);
                            }
                            used_count += __popc(new_key_mask);
                        }

                        // Key publication, value initialization, and used-slot
                        // recording complete before any lane accumulates.
                        __syncwarp();
                        if (destination_slot >= 0) {
                            atomicAdd(
                                &hash_values
                                    [warp_in_block][destination_slot],
                                weight);
                        } else if (valid) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                weight);
                        }
                        __syncwarp();

#ifdef ENABLE_BLOCK_STATS
                        const int warp_probe_attempts = __reduce_add_sync(
                            kFullWarpMask, probe_attempts);
                        const unsigned fallback_mask = __ballot_sync(
                            kFullWarpMask,
                            valid && destination_slot < 0);
                        if (lane == 0 && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kHashProbeAttemptsColumn,
                                warp_probe_attempts);
                            const int fallback_count =
                                __popc(fallback_mask);
                            atomicAdd(
                                block_stats
                                    + kHashFallbackAtomicsColumn,
                                fallback_count);
                            atomicAdd(
                                block_stats + kV4GlobalAtomicsColumn,
                                fallback_count);
                        }
#endif
                    }

                    int flush_count = 0;
                    if constexpr (kHashUsedSlots) {
                        flush_count = used_count;
                        for (int used_index = lane;
                             used_index < used_count;
                             used_index += 32) {
                            const int hash_slot = hash_used_slots
                                [warp_in_block][used_index];
                            const int post =
                                hash_keys[warp_in_block][hash_slot];
                            atomicAdd(
                                psc + b * n_neuron + post,
                                hash_values[warp_in_block][hash_slot]);
                            hash_keys[warp_in_block][hash_slot] =
                                kEmptyHashKey;
                        }
                    } else {
                        const bool occupied =
                            hash_keys[warp_in_block][lane]
                            != kEmptyHashKey;
                        flush_count = __popc(__ballot_sync(
                            kFullWarpMask, occupied));
                        for (int hash_slot = lane;
                             hash_slot < kHashSize;
                             hash_slot += 32) {
                            if (hash_slot >= 32) {
                                const bool extra_occupied =
                                    hash_keys
                                        [warp_in_block][hash_slot]
                                    != kEmptyHashKey;
                                flush_count += __popc(__ballot_sync(
                                    kFullWarpMask, extra_occupied));
                            }
                            const int post =
                                hash_keys[warp_in_block][hash_slot];
                            if (post != kEmptyHashKey) {
                                atomicAdd(
                                    psc + b * n_neuron + post,
                                    hash_values
                                        [warp_in_block][hash_slot]);
                            }
                        }
                    }
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kHashFlushAtomicsColumn,
                            flush_count);
                        atomicAdd(
                            block_stats + kV4GlobalAtomicsColumn,
                            flush_count);
                    }
#endif
                    __syncwarp();
                }
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0
                    && task_used_hash
                    && block_stats != nullptr) {
                    atomicAdd(block_stats + kHashTasksColumn, 1);
                }
#endif
                __syncwarp();
                continue;
#else
                const bool reduce_task =
                    kTileReduceMode == 1
                    || (kTileReduceMode == 2 && task_edges >= 64);
#ifdef ENABLE_BLOCK_STATS
                if (lane == 0 && block_stats != nullptr) {
                    atomicAdd(
                        block_stats + kV4InputEdgesColumn,
                        task_edges);
                    if (reduce_task) {
                        atomicAdd(
                            block_stats + kV4ReduceTasksColumn,
                            1);
                        atomicAdd(
                            block_stats + kV4ReduceEdgesColumn,
                            task_edges);
                    }
                }
#endif
                for (int edge_base = logical_begin;
                     edge_base < logical_end;
                     edge_base += 32) {
                    const int logical_edge = edge_base + lane;
                    const bool valid = logical_edge < logical_end;
                    const int lookup_edge = valid
                        ? logical_edge
                        : logical_end - 1;
                    int low = 0;
                    int high = 31;
#pragma unroll
                    for (int step = 0; step < 5; ++step) {
                        const int middle = (low + high) >> 1;
                        if (packed_prefix[warp_in_block][middle]
                            > lookup_edge) {
                            high = middle;
                        } else {
                            low = middle + 1;
                        }
                    }

                    int post = 0;
                    float weight = 0.0f;
                    if (valid) {
                        const int previous_end = low == 0
                            ? 0
                            : packed_prefix[warp_in_block][low - 1];
                        const int edge =
                            packed_starts[warp_in_block][low]
                            + logical_edge - previous_end;
                        post = graph_indices[edge];
                        weight = graph_weight[edge];
                    }
                    const unsigned valid_mask =
                        __ballot_sync(kFullWarpMask, valid);
                    if (!reduce_task) {
#ifdef ENABLE_BLOCK_STATS
                        if (lane == 0 && block_stats != nullptr) {
                            atomicAdd(
                                block_stats + kV4GlobalAtomicsColumn,
                                __popc(valid_mask));
                        }
#endif
                        if (valid) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                weight);
                        }
                        continue;
                    }

                    unsigned peers = 0;
                    if (valid) {
                        peers = __match_any_sync(valid_mask, post);
                    }
                    const bool is_leader =
                        valid && lane == __ffs(peers) - 1;
                    const unsigned leader_mask =
                        __ballot_sync(kFullWarpMask, is_leader);
#ifdef ENABLE_BLOCK_STATS
                    if (lane == 0 && block_stats != nullptr) {
                        atomicAdd(
                            block_stats + kV4GlobalAtomicsColumn,
                            __popc(leader_mask));
                    }
#endif
                    if (valid) {
                        unsigned remaining = peers;
                        float reduced_weight = 0.0f;
                        while (remaining != 0) {
                            const int source = __ffs(remaining) - 1;
                            reduced_weight += __shfl_sync(
                                peers, weight, source);
                            remaining &= remaining - 1;
                        }
                        if (is_leader) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                reduced_weight);
                        }
                    }
                }
                __syncwarp();
                continue;
#endif
            }

            // Tiny rows stay lane-owned. The remaining medium rows are either
            // consumed as strict adjacent CSR runs or packed into one logical
            // edge stream so scattered masks no longer serialize per row.
            const bool tiny_row =
                lane_fired && lane_degree <= kLaneRowThreshold;
            const unsigned tiny_mask = __ballot_sync(kFullWarpMask, tiny_row);
            const unsigned medium_mask = spike_mask & ~tiny_mask;

            if (tiny_row) {
                for (int edge = lane_edge_start; edge < lane_edge_end; ++edge) {
                    const int post = graph_indices[edge];
                    atomicAdd(
                        psc + b * n_neuron + post,
                        graph_weight[edge]);
                }
            }

            if (medium_mask != 0) {
                const int medium_rows = __popc(medium_mask);
                const unsigned run_starts =
                    medium_mask & ~(medium_mask << 1);
                const int run_count = __popc(run_starts);
                const int medium_degree =
                    (medium_mask & (1u << lane)) != 0 ? lane_degree : 0;
                int inclusive_end = medium_degree;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    const int value = __shfl_up_sync(
                        kFullWarpMask, inclusive_end, offset);
                    if (lane >= offset) {
                        inclusive_end += value;
                    }
                }
                const int medium_edges = __shfl_sync(
                    kFullWarpMask, inclusive_end, 31);
#ifdef ENABLE_BLOCK_HASH
                const bool use_hash =
                    medium_rows >= 2
                    && medium_edges >= kHashMinEdges
                    && medium_edges <= kLegacyHashMaxEdges;
#else
                constexpr bool use_hash = false;
#endif
                if (use_hash) {
                    for (int slot = lane; slot < kHashSize; slot += 32) {
                        hash_keys[warp_in_block][slot] = kEmptyHashKey;
                        hash_values[warp_in_block][slot] = 0.0f;
                    }
                    __syncwarp();
                }
                if (run_count * 2 <= medium_rows) {
                    // P1: adjacent active CSR rows are one physical edge run.
                    unsigned remaining = medium_mask;
                    while (remaining != 0) {
                        const int first_lane = __ffs(remaining) - 1;
                        const unsigned shifted = remaining >> first_lane;
                        const int first_zero = __ffs(~shifted);
                        const int run_length = first_zero == 0
                            ? 32
                            : first_zero - 1;
                        const int end_lane = first_lane + run_length;
                        const int run_begin =
                            graph_indptr[block_start + first_lane];
                        const int run_end =
                            graph_indptr[block_start + end_lane];
                        for (int edge_base = run_begin;
                             edge_base < run_end;
                             edge_base += 32) {
                            const int edge = edge_base + lane;
                            if (edge < run_end) {
                                const int post = graph_indices[edge];
                                const float weight = graph_weight[edge];
                                if (use_hash) {
                                    int slot = static_cast<int>(
                                        (static_cast<unsigned>(post)
                                         * 2654435761u)
                                        & (kHashSize - 1));
                                    bool inserted = false;
#pragma unroll
                                    for (int probe = 0;
                                         probe < kHashMaxProbe;
                                         ++probe) {
                                        const int old = atomicCAS(
                                            &hash_keys[warp_in_block][slot],
                                            kEmptyHashKey,
                                            post);
                                        if (old == kEmptyHashKey
                                            || old == post) {
                                            atomicAdd(
                                                &hash_values
                                                    [warp_in_block][slot],
                                                weight);
                                            inserted = true;
                                            break;
                                        }
                                        slot = (slot + 1) & (kHashSize - 1);
                                    }
                                    if (!inserted) {
                                        atomicAdd(
                                            psc + b * n_neuron + post,
                                            weight);
                                    }
                                } else {
                                    atomicAdd(
                                        psc + b * n_neuron + post,
                                        weight);
                                }
                            }
                        }
                        const unsigned run_mask = run_length == 32
                            ? kFullWarpMask
                            : ((1u << run_length) - 1u) << first_lane;
                        remaining &= ~run_mask;
                    }
                } else {
                    // P3: prefix-pack scattered rows into one logical stream.
                    packed_prefix[warp_in_block][lane] = inclusive_end;
                    packed_starts[warp_in_block][lane] = lane_edge_start;
                    __syncwarp();
                    const int total_edges =
                        packed_prefix[warp_in_block][31];
                    for (int edge_base = 0;
                         edge_base < total_edges;
                         edge_base += 32) {
                        const int logical_edge = edge_base + lane;
                        const bool valid = logical_edge < total_edges;
                        const int lookup_edge = valid
                            ? logical_edge
                            : total_edges - 1;
                        int low = 0;
                        int high = 31;
#pragma unroll
                        for (int step = 0; step < 5; ++step) {
                            const int middle = (low + high) >> 1;
                            if (packed_prefix[warp_in_block][middle]
                                > lookup_edge) {
                                high = middle;
                            } else {
                                low = middle + 1;
                            }
                        }
                        if (valid) {
                            const int previous_end = low == 0
                                ? 0
                                : packed_prefix[warp_in_block][low - 1];
                            const int edge =
                                packed_starts[warp_in_block][low]
                                + logical_edge - previous_end;
                            const int post = graph_indices[edge];
                            const float weight = graph_weight[edge];
                            if (use_hash) {
                                int slot = static_cast<int>(
                                    (static_cast<unsigned>(post)
                                     * 2654435761u)
                                    & (kHashSize - 1));
                                bool inserted = false;
#pragma unroll
                                for (int probe = 0;
                                     probe < kHashMaxProbe;
                                     ++probe) {
                                    const int old = atomicCAS(
                                        &hash_keys[warp_in_block][slot],
                                        kEmptyHashKey,
                                        post);
                                    if (old == kEmptyHashKey || old == post) {
                                        atomicAdd(
                                            &hash_values
                                                [warp_in_block][slot],
                                            weight);
                                        inserted = true;
                                        break;
                                    }
                                    slot = (slot + 1) & (kHashSize - 1);
                                }
                                if (!inserted) {
                                    atomicAdd(
                                        psc + b * n_neuron + post,
                                        weight);
                                }
                            } else {
                                atomicAdd(
                                    psc + b * n_neuron + post,
                                    weight);
                            }
                        }
                    }
                    __syncwarp();
                }
                if (use_hash) {
                    __syncwarp();
                    for (int slot = lane; slot < kHashSize; slot += 32) {
                        const int post = hash_keys[warp_in_block][slot];
                        if (post != kEmptyHashKey) {
                            atomicAdd(
                                psc + b * n_neuron + post,
                                hash_values[warp_in_block][slot]);
                        }
                    }
                    __syncwarp();
                }
            }
        }
        grid.sync();
    }

    if (is_update_block && resident_v) {
        for (int n = neuron_begin + threadIdx.x;
             n < neuron_end;
             n += blockDim.x) {
            v[n] = shared_v[n - neuron_begin];
        }
    }
}

}  // namespace

void launch_persistent_snn_spike_block_kernel(
    const int* event_offsets,
    const int* event_indices,
    const float* event_values,
    bool has_event_values,
    const int* graph_indptr,
    const int* graph_indices,
    const float* graph_weight,
    float* v,
    float* psc,
    float* dense_spikes,
    float* input_current,
    int* task_queue_batch,
    int* task_queue_start,
    int* task_queue_end_or_mask,
    int* task_counts,
    int* work_counters,
    int* event_counts,
    int* event_indices_full,
    int* block_stats,
    bool return_dense,
    bool return_events,
    int t_steps,
    int batch_size,
    int n_neuron,
    int queue_capacity,
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m,
    int grid_dim,
    int block_dim,
    cudaStream_t stream) {
    void* args[] = {
        &event_offsets,
        &event_indices,
        &event_values,
        &has_event_values,
        &graph_indptr,
        &graph_indices,
        &graph_weight,
        &v,
        &psc,
        &dense_spikes,
        &input_current,
        &task_queue_batch,
        &task_queue_start,
        &task_queue_end_or_mask,
        &task_counts,
        &work_counters,
        &event_counts,
        &event_indices_full,
        &block_stats,
        &return_events,
        &t_steps,
        &batch_size,
        &n_neuron,
        &queue_capacity,
        &dt,
        &tau_mem,
        &tau_syn,
        &v_threshold,
        &v_reset,
        &c_m,
    };
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_spike_block_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_spike_block_kernel<false>);
    cudaLaunchCooperativeKernel(
        kernel,
        grid_dim,
        block_dim,
        args,
        0,
        stream);
#ifdef ENABLE_BLOCK_STATS
    if (block_stats != nullptr) {
        const int blocks_per_batch = (n_neuron + 31) / 32;
        const int record_count = t_steps * batch_size * blocks_per_batch;
        constexpr int kStatsThreads = 256;
        collect_block_stats_kernel<<<
            (record_count + kStatsThreads - 1) / kStatsThreads,
            kStatsThreads,
            0,
            stream>>>(
            dense_spikes,
            graph_indptr,
            block_stats,
            t_steps,
            batch_size,
            n_neuron,
            blocks_per_batch);
    }
#else
    (void)block_stats;
#endif
}

int persistent_snn_spike_block_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events) {
    int active_blocks = 0;
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_spike_block_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_spike_block_kernel<false>);
    const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        kernel,
        block_dim,
        0);
    if (error != cudaSuccess) {
        return 0;
    }
    return active_blocks;
}
