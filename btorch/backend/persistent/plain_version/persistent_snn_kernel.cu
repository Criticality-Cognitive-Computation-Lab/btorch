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
    float* __restrict__ psc_h,
    float* __restrict__ asc,
    float* __restrict__ refractory,
    float* __restrict__ dense_spikes,
    float* __restrict__ input_current,
    int* __restrict__ spike_queue_batch,
    int* __restrict__ spike_queue_pre,
    int* __restrict__ spike_count,
    int* __restrict__ work_counter,
    int* __restrict__ event_counts,
    int* __restrict__ event_indices_full,
    int* __restrict__ overflow,
    bool return_events,
    int t_steps,
    int batch_size,
    int n_neuron,
    float dt,
    float tau,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float v_rest,
    float c_m,
    float tau_ref,
    float k0,
    float k1,
    float asc_amp0,
    float asc_amp1,
    float psc_g_max) {
    cg::grid_group grid = cg::this_grid();
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int n_cells = batch_size * n_neuron;
    const int queue_capacity = n_cells;
    const float syn_decay = expf(-dt / tau_syn);
    const float mem_decay = expf(-dt / tau);
    const float asc_decay0 = expf(-dt * k0);
    const float asc_decay1 = expf(-dt * k1);
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
            psc[cell] = syn_decay * psc[cell] + (1.0f - syn_decay) * psc_h[cell];
            psc_h[cell] = syn_decay * psc_h[cell];

            const int asc_base = (b * n_neuron + n) * 2;
            const float asc0_old = asc[asc_base];
            const float asc1_old = asc[asc_base + 1];
            const float asc0_next = asc0_old * asc_decay0;
            const float asc1_next = asc1_old * asc_decay1;
            asc[asc_base] = asc0_next;
            asc[asc_base + 1] = asc1_next;

            const float current = psc[cell] + input_current[cell];
            const float current_sum = current + asc0_old + asc1_old;
            const float v_inf = v_rest + tau * current_sum / c_m;
            const float v_pre = v_inf + (v[cell] - v_inf) * mem_decay;
            const bool can_fire = refractory[cell] <= 0.0f;
            const bool fired = can_fire && v_pre >= v_threshold;
            const float spike = fired ? 1.0f : 0.0f;
            v[cell] = v_pre - reset_delta * spike;
            refractory[cell] = fmaxf(refractory[cell] - dt, 0.0f);
            if (fired) {
                refractory[cell] = fmaxf(tau_ref - dt, 0.0f);
                asc[asc_base] += asc_amp0;
                asc[asc_base + 1] += asc_amp1;
            }
            dense_spikes[(t * batch_size + b) * n_neuron + n] = spike;

            if (fired) {
                const int task = atomicAdd(spike_count, 1);
                if (task < queue_capacity) {
                    spike_queue_batch[task] = b;
                    spike_queue_pre[task] = n;
                } else {
                    atomicExch(overflow, 1);
                }

                if (return_events) {
                    const int bucket = t * batch_size + b;
                    const int rank = atomicAdd(event_counts + bucket, 1);
                    if (rank < n_neuron) {
                        event_indices_full[bucket * n_neuron + rank] = n;
                    } else {
                        atomicExch(overflow, 1);
                    }
                }
            }
        }
        grid.sync();

        while (true) {
            const int task = atomicAdd(work_counter, 1);
            const int count = *spike_count;
            if (task >= count || task >= queue_capacity) {
                break;
            }
            const int b = spike_queue_batch[task];
            const int pre = spike_queue_pre[task];
            const int start = graph_indptr[pre];
            const int end = graph_indptr[pre + 1];
            for (int edge = start; edge < end; ++edge) {
                const int post = graph_indices[edge];
                if (post >= 0 && post < n_neuron) {
                    atomicAdd(
                        psc_h + b * n_neuron + post,
                        psc_g_max * graph_weight[edge]);
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
    float* psc_h,
    float* asc,
    float* refractory,
    float* dense_spikes,
    float* input_current,
    int* spike_queue_batch,
    int* spike_queue_pre,
    int* spike_count,
    int* work_counter,
    int* event_counts,
    int* event_indices_full,
    int* overflow,
    bool return_events,
    int t_steps,
    int batch_size,
    int n_neuron,
    float dt,
    float tau,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float v_rest,
    float c_m,
    float tau_ref,
    float k0,
    float k1,
    float asc_amp0,
    float asc_amp1,
    float psc_g_max,
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
        &psc_h,
        &asc,
        &refractory,
        &dense_spikes,
        &input_current,
        &spike_queue_batch,
        &spike_queue_pre,
        &spike_count,
        &work_counter,
        &event_counts,
        &event_indices_full,
        &overflow,
        &return_events,
        &t_steps,
        &batch_size,
        &n_neuron,
        &dt,
        &tau,
        &tau_syn,
        &v_threshold,
        &v_reset,
        &v_rest,
        &c_m,
        &tau_ref,
        &k0,
        &k1,
        &asc_amp0,
        &asc_amp1,
        &psc_g_max,
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
