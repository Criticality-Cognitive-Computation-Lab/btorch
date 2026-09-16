#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kEdgesPerTask = 1024;

#ifdef ENABLE_PIPELINE_TIMING
constexpr int kPipelineTimingColumns = 5;

enum PipelineTimestamp : int {
    kTimestampBegin = 0,
    kTimestampFirstPublish = 1,
    kTimestampFirstConsume = 2,
    kTimestampUpdateDone = 3,
    kTimestampPipelineDone = 4,
};

__device__ __forceinline__ unsigned long long global_timestamp_ns() {
    unsigned long long timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}
#endif

template <bool ReturnDense>
__global__ void persistent_snn_kernel(
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
    int* __restrict__ spike_queue_batch,
    int* __restrict__ spike_queue_edge_start,
    int* __restrict__ spike_queue_edge_end,
    int* __restrict__ spike_count,
    int* __restrict__ work_counter,
    unsigned long long* __restrict__ timing_stats,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
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
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;

    for (int t = 0; t < t_steps; ++t) {
        if (global_tid == 0) {
            *spike_count = 0;
            *work_counter = 0;
        }
        for (int b = 0; b < batch_size; ++b) {
            const int bucket = t * batch_size + b;
            const int start = event_offsets[bucket];
            const int end = event_offsets[bucket + 1];
            for (int event = start + global_tid; event < end; event += stride) {
                const int pre = event_indices[event];
                if (pre >= 0 && pre < n_neuron) {
                    const float value = has_event_values ? event_values[event] : 1.0f;
                    atomicAdd(input_current + b * n_neuron + pre, value);
                }
            }
        }
#ifdef ENABLE_PIPELINE_TIMING
        if (global_tid == 0) {
            timing_stats[t * kPipelineTimingColumns + kTimestampBegin] =
                global_timestamp_ns();
        }
#endif
        grid.sync();

        for (int b = 0; b < batch_size; ++b) {
            const int batch_base = b * n_neuron;
            for (int n = global_tid; n < n_neuron; n += stride) {
                const int cell = batch_base + n;
                const float current = psc[cell] + input_current[cell];
                input_current[cell] = 0.0f;
                const float v_pre = v[cell] +
                    dt * (-(v[cell] - v_reset) / tau_mem + current / c_m);
                const bool fired = v_pre >= v_threshold;
                const float spike = fired ? 1.0f : 0.0f;
                v[cell] = v_pre - reset_delta * spike;
                if constexpr (ReturnDense) {
                    dense_spikes[(t * batch_size + b) * n_neuron + n] = spike;
                }
                // Fold the PSC decay into this loop. Each thread owns
                // psc[cell], and current uses the old psc before recurrent
                // fanout atomicAdds update it.
                psc[cell] *= decay;

                if (fired) {
                    const int edge_start = graph_indptr[n];
                    const int edge_end = graph_indptr[n + 1];
                    const int task_count = (edge_end - edge_start +
                                            kEdgesPerTask - 1) /
                        kEdgesPerTask;
                    const int first_task = atomicAdd(spike_count, task_count);
                    if (first_task + task_count <= queue_capacity) {
                        for (int slice = 0; slice < task_count; ++slice) {
                            const int task = first_task + slice;
                            const int slice_start =
                                edge_start + slice * kEdgesPerTask;
                            spike_queue_batch[task] = b;
                            spike_queue_edge_start[task] = slice_start;
                            spike_queue_edge_end[task] =
                                min(slice_start + kEdgesPerTask, edge_end);
                        }
                    }

                    if (return_events) {
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
#ifdef ENABLE_PIPELINE_TIMING
        if (global_tid == 0) {
            const unsigned long long timestamp = global_timestamp_ns();
            timing_stats[
                t * kPipelineTimingColumns + kTimestampFirstPublish] =
                timestamp;
            timing_stats[
                t * kPipelineTimingColumns + kTimestampFirstConsume] =
                timestamp;
            timing_stats[t * kPipelineTimingColumns + kTimestampUpdateDone] =
                timestamp;
        }
#endif

        // Long rows were already split into bounded edge ranges when queued.
        // One warp claims one task at a time so skewed row lengths remain
        // dynamically balanced across the cooperative grid.
        const int lane = global_tid & 31;
        const int count = *spike_count;
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counter, 1);
            }
            task = __shfl_sync(0xffffffffu, task, 0);
            if (task >= count || task >= queue_capacity) {
                break;
            }
            const int b = spike_queue_batch[task];
            const int start = spike_queue_edge_start[task];
            const int end = spike_queue_edge_end[task];
            for (int edge = start + lane; edge < end; edge += 32) {
                const int post = graph_indices[edge];
                atomicAdd(psc + b * n_neuron + post, graph_weight[edge]);
            }
        }
        grid.sync();
#ifdef ENABLE_PIPELINE_TIMING
        if (global_tid == 0) {
            timing_stats[
                t * kPipelineTimingColumns + kTimestampPipelineDone] =
                global_timestamp_ns();
        }
#endif
    }
}

__global__ void compact_event_indices_kernel(
    const int* __restrict__ event_counts,
    const int* __restrict__ event_offsets,
    const int* __restrict__ event_indices_full,
    int* __restrict__ event_indices,
    int n_buckets,
    int n_neuron) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int bucket = global_tid; bucket < n_buckets; bucket += stride) {
        const int count = event_counts[bucket];
        const int src_base = bucket * n_neuron;
        const int dst_base = event_offsets[bucket];
        for (int i = 0; i < count; ++i) {
            event_indices[dst_base + i] = event_indices_full[src_base + i];
        }
    }
}

}  // namespace

void launch_persistent_snn_kernel(
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
    int* spike_queue_batch,
    int* spike_queue_edge_start,
    int* spike_queue_edge_end,
    int* spike_count,
    int* work_counter,
    unsigned long long* timing_stats,
    int* event_counts,
    int* event_indices_full,
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
        &spike_queue_batch,
        &spike_queue_edge_start,
        &spike_queue_edge_end,
        &spike_count,
        &work_counter,
        &timing_stats,
        &event_counts,
        &event_indices_full,
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
        ? reinterpret_cast<void*>(persistent_snn_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_kernel<false>);
    cudaLaunchCooperativeKernel(
        kernel,
        grid_dim,
        block_dim,
        args,
        0,
        stream);
}

int persistent_snn_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events) {
    int active_blocks = 0;
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_kernel<false>);
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

void launch_compact_event_indices_kernel(
    const int* event_counts,
    const int* event_offsets,
    const int* event_indices_full,
    int* event_indices,
    int n_buckets,
    int n_neuron,
    int grid_dim,
    int block_dim,
    cudaStream_t stream) {
    compact_event_indices_kernel<<<grid_dim, block_dim, 0, stream>>>(
        event_counts,
        event_offsets,
        event_indices_full,
        event_indices,
        n_buckets,
        n_neuron);
}
