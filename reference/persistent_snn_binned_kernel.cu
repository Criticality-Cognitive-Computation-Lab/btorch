#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kLowTasksPerWarp = 4;
constexpr int kLowSubwarpSize = 8;
constexpr int kEdgesPerTask = 1024;

template <bool ReturnDense>
__global__ void persistent_snn_binned_kernel(
    const int* __restrict__ event_offsets,
    const int* __restrict__ event_indices,
    const float* __restrict__ event_values,
    bool has_event_values,
    const int* __restrict__ graph_indptr,
    const int* __restrict__ graph_indices,
    const float* __restrict__ graph_weight,
    const int* __restrict__ graph_high_fanout,
    float* __restrict__ v,
    float* __restrict__ psc,
    float* __restrict__ dense_spikes,
    float* __restrict__ input_current,
    int* __restrict__ task_queue_batch,
    int* __restrict__ task_queue_edge_start,
    int* __restrict__ task_queue_edge_end,
    int* __restrict__ task_counts,
    int* __restrict__ work_counters,
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
                psc[cell] *= decay;

                if (fired) {
                    if (graph_high_fanout[n] != 0) {
                        const int edge_start = graph_indptr[n];
                        const int edge_end = graph_indptr[n + 1];
                        const int task_count = (edge_end - edge_start +
                                                kEdgesPerTask - 1) /
                            kEdgesPerTask;
                        const int first_task =
                            atomicAdd(task_counts, task_count);
                        for (int segment = 0; segment < task_count; ++segment) {
                            const int task = first_task + segment;
                            const int segment_start =
                                edge_start + segment * kEdgesPerTask;
                            task_queue_batch[task] = b;
                            task_queue_edge_start[task] = segment_start;
                            task_queue_edge_end[task] =
                                min(segment_start + kEdgesPerTask, edge_end);
                        }
                    } else {
                        const int task = atomicAdd(task_counts + 1, 1);
                        const int slot = queue_capacity - 1 - task;
                        task_queue_batch[slot] = b;
                        task_queue_edge_start[slot] = n;
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

        const int lane = global_tid & 31;
        const int high_count = task_counts[0];
        while (true) {
            int task = 0;
            if (lane == 0) {
                task = atomicAdd(work_counters, 1);
            }
            task = __shfl_sync(0xffffffffu, task, 0);
            if (task >= high_count || task >= queue_capacity) {
                break;
            }
            const int b = task_queue_batch[task];
            const int start = task_queue_edge_start[task];
            const int end = task_queue_edge_end[task];
            for (int edge = start + lane; edge < end; edge += 32) {
                const int post = graph_indices[edge];
                atomicAdd(psc + b * n_neuron + post, graph_weight[edge]);
            }
        }

        const int low_count = task_counts[1];
        const int subwarp = lane / kLowSubwarpSize;
        const int sublane = lane - subwarp * kLowSubwarpSize;
        while (true) {
            int base_task = 0;
            if (lane == 0) {
                base_task = atomicAdd(work_counters + 1, kLowTasksPerWarp);
            }
            base_task = __shfl_sync(0xffffffffu, base_task, 0);
            if (base_task >= low_count || base_task >= queue_capacity) {
                break;
            }
            const int task = base_task + subwarp;
            if (task < low_count && task < queue_capacity) {
                const int slot = queue_capacity - 1 - task;
                const int b = task_queue_batch[slot];
                const int pre = task_queue_edge_start[slot];
                const int start = graph_indptr[pre];
                const int end = graph_indptr[pre + 1];
                for (int edge = start + sublane; edge < end;
                     edge += kLowSubwarpSize) {
                    const int post = graph_indices[edge];
                    atomicAdd(psc + b * n_neuron + post, graph_weight[edge]);
                }
            }
        }
        grid.sync();
    }
}

}  // namespace

void launch_persistent_snn_binned_kernel(
    const int* event_offsets,
    const int* event_indices,
    const float* event_values,
    bool has_event_values,
    const int* graph_indptr,
    const int* graph_indices,
    const float* graph_weight,
    const int* graph_high_fanout,
    float* v,
    float* psc,
    float* dense_spikes,
    float* input_current,
    int* task_queue_batch,
    int* task_queue_edge_start,
    int* task_queue_edge_end,
    int* task_counts,
    int* work_counters,
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
        &graph_high_fanout,
        &v,
        &psc,
        &dense_spikes,
        &input_current,
        &task_queue_batch,
        &task_queue_edge_start,
        &task_queue_edge_end,
        &task_counts,
        &work_counters,
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
        ? reinterpret_cast<void*>(persistent_snn_binned_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_binned_kernel<false>);
    cudaLaunchCooperativeKernel(
        kernel,
        grid_dim,
        block_dim,
        args,
        0,
        stream);
}

int persistent_snn_binned_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events) {
    int active_blocks = 0;
    const void* kernel = return_dense
        ? reinterpret_cast<void*>(persistent_snn_binned_kernel<true>)
        : reinterpret_cast<void*>(persistent_snn_binned_kernel<false>);
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
