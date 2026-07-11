#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace cg = cooperative_groups;

namespace {

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
    int* __restrict__ spike_queue_pre,
    int* __restrict__ spike_count,
    int* __restrict__ work_counter,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
    int* __restrict__ overflow,
    int t_steps,
    int batch_size,
    int n_neuron,
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m) {
    cg::grid_group grid = cg::this_grid();
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int n_cells = batch_size * n_neuron;
    const int queue_capacity = n_cells;
    const float decay = expf(-dt / tau_syn);
    const float reset_delta = v_threshold - v_reset;

    for (int t = 0; t < t_steps; ++t) {
        if (global_tid == 0) {
            *spike_count = 0;
            *work_counter = 0;
        }
        for (int i = global_tid; i < n_cells; i += stride) {
            input_current[i] = 0.0f;
        }
        grid.sync();

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
        grid.sync();

        for (int cell = global_tid; cell < n_cells; cell += stride) {
            const int b = cell / n_neuron;
            const int n = cell - b * n_neuron;
            const float current = psc[cell] + input_current[cell];
            const float v_pre =
                v[cell] + dt * (-(v[cell] - v_reset) / tau_mem + current / c_m);
            const bool fired = v_pre >= v_threshold;
            const float spike = fired ? 1.0f : 0.0f;
            v[cell] = v_pre - reset_delta * spike;
            dense_spikes[(t * batch_size + b) * n_neuron + n] = spike;
            // Fold the PSC decay into this loop. Each thread owns psc[cell]
            // (grid-stride, no cross-thread aliasing), and the reference order
            // is: current uses the *old* psc; then psc = psc*decay + recurrent.
            // We already read psc[cell] into `current` above, so decaying it
            // here -- before the fanout phase atomicAdds the recurrent term --
            // reproduces that order exactly while removing a separate decay
            // pass over n_cells and its grid.sync() every timestep.
            psc[cell] *= decay;

            if (fired) {
                const int task = atomicAdd(spike_count, 1);
                if (task < queue_capacity) {
                    spike_queue_batch[task] = b;
                    spike_queue_pre[task] = n;
                } else {
                    atomicExch(overflow, 1);
                }

                const int bucket = t * batch_size + b;
                const int rank = atomicAdd(event_counts + bucket, 1);
                if (rank < n_neuron) {
                    event_indices_full[bucket * n_neuron + rank] = n;
                } else {
                    atomicExch(overflow, 1);
                }
            }
        }
        grid.sync();

        // Task3: recurrent prespan fanout -- warp-per-neuron work stealing.
        //
        // The original scheme was one-thread-per-neuron: every one of the
        // ~gridDim*blockDim grid threads did atomicAdd(work_counter, 1) in
        // this loop -- a contention storm on a single global int, even though
        // only spike_count neurons have work -- and a single thread then
        // walked that neuron's entire edge list serially. For sparse activity
        // that left ~99% of threads idle while they all hammered one atomic.
        //
        // Here one *warp* cooperatively processes one spiking neuron: lane 0
        // claims the next task with a single atomicAdd (warp-aggregated, so
        // work_counter sees 32x fewer atomics), broadcasts it to the warp,
        // and all 32 lanes walk the neuron's CSR edge list in parallel. Work
        // stealing (vs. a static grid-stride split) keeps this balanced for
        // the production graph's highly skewed fanout (avg ~177, P99 ~1911,
        // max ~4165 per plan.md): a warp that finishes a low-fanout neuron
        // immediately grabs the next task instead of idling.
        //
        // blockDim.x is a multiple of 32, so every warp is full and the
        // 0xffffffff mask is valid; `task` is broadcast so the whole warp
        // takes the same break decision and stays converged into grid.sync().
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
            const int pre = spike_queue_pre[task];
            const int start = graph_indptr[pre];
            const int end = graph_indptr[pre + 1];
            for (int edge = start + lane; edge < end; edge += 32) {
                const int post = graph_indices[edge];
                if (post >= 0 && post < n_neuron) {
                    atomicAdd(psc + b * n_neuron + post, graph_weight[edge]);
                }
            }
        }
        grid.sync();
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
    int* spike_queue_pre,
    int* spike_count,
    int* work_counter,
    int* event_counts,
    int* event_indices_full,
    int* overflow,
    int t_steps,
    int batch_size,
    int n_neuron,
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
        &spike_queue_pre,
        &spike_count,
        &work_counter,
        &event_counts,
        &event_indices_full,
        &overflow,
        &t_steps,
        &batch_size,
        &n_neuron,
        &dt,
        &tau_mem,
        &tau_syn,
        &v_threshold,
        &v_reset,
        &c_m,
    };
    cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistent_snn_kernel),
        grid_dim,
        block_dim,
        args,
        0,
        stream);
}

int persistent_snn_max_active_blocks_per_sm(int block_dim) {
    int active_blocks = 0;
    const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        persistent_snn_kernel,
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
