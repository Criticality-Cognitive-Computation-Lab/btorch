#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

constexpr int kThreadsPerBlock = 256;

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
    cudaStream_t stream);

void launch_compact_event_indices_kernel(
    const int* event_counts,
    const int* event_offsets,
    const int* event_indices_full,
    int* event_indices,
    int n_buckets,
    int n_neuron,
    int grid_dim,
    int block_dim,
    cudaStream_t stream);

int persistent_snn_max_active_blocks_per_sm(int block_dim);

void check_cuda(cudaError_t error, const char* message) {
    TORCH_CHECK(error == cudaSuccess, message, ": ", cudaGetErrorString(error));
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    torch::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype.");
}

void check_same_device(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name,
        " must be on the same CUDA device as v.");
}

void check_shape(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
    TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " must match reference shape.");
}

int cooperative_grid_dim(int block_dim) {
    int device = -1;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties failed");
    TORCH_CHECK(
        prop.cooperativeLaunch,
        "persistent SNN requires CUDA cooperative launch support.");

    const int active_blocks = persistent_snn_max_active_blocks_per_sm(block_dim);
    TORCH_CHECK(active_blocks > 0, "persistent SNN kernel has zero occupancy.");
    return active_blocks * prop.multiProcessorCount;
}

void validate_persistent_inputs(
    const torch::Tensor& event_offsets,
    const torch::Tensor& event_indices,
    const torch::Tensor& event_values,
    bool has_event_values,
    const torch::Tensor& graph_indptr,
    const torch::Tensor& graph_indices,
    const torch::Tensor& graph_weight,
    const torch::Tensor& graph_delay,
    bool has_delay,
    const torch::Tensor& v,
    const torch::Tensor& psc,
    const torch::Tensor& psc_h,
    const torch::Tensor& asc,
    const torch::Tensor& refractory,
    double tau,
    double tau_syn,
    double c_m,
    double tau_ref,
    double k0,
    double k1) {
    check_cuda_tensor(event_offsets, "event_offsets", torch::kInt32);
    check_cuda_tensor(event_indices, "event_indices", torch::kInt32);
    check_cuda_tensor(graph_indptr, "graph_indptr", torch::kInt32);
    check_cuda_tensor(graph_indices, "graph_indices", torch::kInt32);
    check_cuda_tensor(graph_weight, "graph_weight", torch::kFloat32);
    check_cuda_tensor(v, "v", torch::kFloat32);
    check_cuda_tensor(psc, "psc", torch::kFloat32);
    check_cuda_tensor(psc_h, "psc_h", torch::kFloat32);
    check_cuda_tensor(asc, "asc", torch::kFloat32);
    check_cuda_tensor(refractory, "refractory", torch::kFloat32);
    check_same_device(event_offsets, v, "event_offsets");
    check_same_device(event_indices, v, "event_indices");
    check_same_device(graph_indptr, v, "graph_indptr");
    check_same_device(graph_indices, v, "graph_indices");
    check_same_device(graph_weight, v, "graph_weight");
    check_same_device(psc, v, "psc");
    check_same_device(psc_h, v, "psc_h");
    check_same_device(asc, v, "asc");
    check_same_device(refractory, v, "refractory");
    if (has_event_values) {
        check_cuda_tensor(event_values, "event_values", torch::kFloat32);
        check_same_device(event_values, v, "event_values");
        TORCH_CHECK(
            event_values.sizes() == event_indices.sizes(),
            "event_values must match event_indices shape.");
    }
    if (has_delay) {
        check_cuda_tensor(graph_delay, "graph_delay", torch::kInt32);
        check_same_device(graph_delay, v, "graph_delay");
        TORCH_CHECK(
            graph_delay.numel() == graph_indices.numel(),
            "graph_delay must match graph_indices shape.");
    }

    if (has_delay) {
        TORCH_CHECK(
            graph_delay.eq(0).all().item<bool>(),
            "persistent SNN v1 does not support nonzero delay.");
    }

    TORCH_CHECK(v.dim() == 2, "v must have shape (B, N).");
    TORCH_CHECK(psc.sizes() == v.sizes(), "psc must match v shape.");
    TORCH_CHECK(psc_h.sizes() == v.sizes(), "psc_h must match v shape.");
    TORCH_CHECK(refractory.sizes() == v.sizes(), "refractory must match v shape.");
    const auto batch_size = static_cast<int>(v.size(0));
    const auto n_neuron = static_cast<int>(v.size(1));
    TORCH_CHECK(
        asc.dim() == 3 && asc.size(0) == batch_size && asc.size(1) == n_neuron &&
            asc.size(2) == 2,
        "asc must have shape (B, N, 2).");
    TORCH_CHECK(batch_size > 0 && n_neuron > 0, "B and N must be positive.");
    TORCH_CHECK(
        graph_indptr.numel() == n_neuron + 1,
        "graph_indptr must have shape (N + 1,).");
    TORCH_CHECK(
        graph_indices.numel() == graph_weight.numel(),
        "graph_indices and graph_weight must match.");
    TORCH_CHECK(event_offsets.dim() == 1, "event_offsets must be 1D.");
    TORCH_CHECK(event_indices.dim() == 1, "event_indices must be 1D.");
    TORCH_CHECK(
        event_offsets.numel() >= 2,
        "event_offsets must contain at least one bucket.");
    TORCH_CHECK(
        (event_offsets.numel() - 1) % batch_size == 0,
        "event_offsets bucket count must be divisible by batch size.");
    TORCH_CHECK(
        tau > 0.0 && tau_syn > 0.0 && c_m > 0.0 && tau_ref >= 0.0 &&
            k0 > 0.0 && k1 > 0.0,
        "tau, tau_syn, c_m, tau_ref, and k values must be valid.");
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
persistent_snn_forward_cuda(
    torch::Tensor event_offsets,
    torch::Tensor event_indices,
    torch::Tensor event_values,
    bool has_event_values,
    torch::Tensor graph_indptr,
    torch::Tensor graph_indices,
    torch::Tensor graph_weight,
    torch::Tensor graph_delay,
    bool has_delay,
    torch::Tensor v,
    torch::Tensor psc,
    torch::Tensor psc_h,
    torch::Tensor asc,
    torch::Tensor refractory,
    double dt,
    double tau,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double v_rest,
    double c_m,
    double tau_ref,
    double k0,
    double k1,
    double asc_amp0,
    double asc_amp1,
    double psc_g_max,
    bool hard_reset,
    bool return_events) {
    TORCH_CHECK(!hard_reset, "persistent SNN v1 only supports soft reset.");

    c10::cuda::CUDAGuard guard(v.device());
    validate_persistent_inputs(
        event_offsets,
        event_indices,
        event_values,
        has_event_values,
        graph_indptr,
        graph_indices,
        graph_weight,
        graph_delay,
        has_delay,
        v,
        psc,
        psc_h,
        asc,
        refractory,
        tau,
        tau_syn,
        c_m,
        tau_ref,
        k0,
        k1);
    const auto batch_size = static_cast<int>(v.size(0));
    const auto n_neuron = static_cast<int>(v.size(1));
    const auto t_steps =
        static_cast<int>((event_offsets.numel() - 1) / batch_size);
    TORCH_CHECK(t_steps > 0, "T must be positive.");

    const auto options_f = v.options();
    const auto options_i = event_offsets.options();
    auto v_out = v.clone();
    auto psc_out = psc.clone();
    auto psc_h_out = psc_h.clone();
    auto asc_out = asc.clone();
    auto refractory_out = refractory.clone();
    auto dense_spikes = torch::empty({t_steps, batch_size, n_neuron}, options_f);
    auto input_current = torch::empty({batch_size, n_neuron}, options_f);
    auto spike_queue_batch = torch::empty({batch_size * n_neuron}, options_i);
    auto spike_queue_pre = torch::empty({batch_size * n_neuron}, options_i);
    auto spike_count = torch::zeros({1}, options_i);
    auto work_counter = torch::zeros({1}, options_i);
    auto event_counts = return_events ? torch::zeros({t_steps * batch_size}, options_i)
                                      : torch::empty({0}, options_i);
    auto event_indices_full =
        return_events ? torch::empty({t_steps * batch_size * n_neuron}, options_i)
                      : torch::empty({0}, options_i);
    auto overflow = torch::zeros({1}, options_i);

    const int grid_dim = cooperative_grid_dim(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    launch_persistent_snn_kernel(
        event_offsets.data_ptr<int>(),
        event_indices.data_ptr<int>(),
        has_event_values ? event_values.data_ptr<float>() : nullptr,
        has_event_values,
        graph_indptr.data_ptr<int>(),
        graph_indices.data_ptr<int>(),
        graph_weight.data_ptr<float>(),
        v_out.data_ptr<float>(),
        psc_out.data_ptr<float>(),
        psc_h_out.data_ptr<float>(),
        asc_out.data_ptr<float>(),
        refractory_out.data_ptr<float>(),
        dense_spikes.data_ptr<float>(),
        input_current.data_ptr<float>(),
        spike_queue_batch.data_ptr<int>(),
        spike_queue_pre.data_ptr<int>(),
        spike_count.data_ptr<int>(),
        work_counter.data_ptr<int>(),
        event_counts.data_ptr<int>(),
        event_indices_full.data_ptr<int>(),
        overflow.data_ptr<int>(),
        return_events,
        t_steps,
        batch_size,
        n_neuron,
        static_cast<float>(dt),
        static_cast<float>(tau),
        static_cast<float>(tau_syn),
        static_cast<float>(v_threshold),
        static_cast<float>(v_reset),
        static_cast<float>(v_rest),
        static_cast<float>(c_m),
        static_cast<float>(tau_ref),
        static_cast<float>(k0),
        static_cast<float>(k1),
        static_cast<float>(asc_amp0),
        static_cast<float>(asc_amp1),
        static_cast<float>(psc_g_max),
        grid_dim,
        kThreadsPerBlock,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto event_offsets_out = torch::empty({0}, options_i);
    auto event_indices_out = torch::empty({0}, options_i);
    if (return_events) {
        event_offsets_out = torch::empty({t_steps * batch_size + 1}, options_i);
        event_offsets_out[0].zero_();
        event_offsets_out.slice(0, 1).copy_(torch::cumsum(event_counts, 0));
        const int total_spikes =
            event_offsets_out[event_offsets_out.numel() - 1].item<int>();
        event_indices_out = torch::empty({total_spikes}, options_i);
        const int compact_grid = std::min(
            grid_dim,
            (t_steps * batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
        if (total_spikes > 0) {
            launch_compact_event_indices_kernel(
                event_counts.data_ptr<int>(),
                event_offsets_out.data_ptr<int>(),
                event_indices_full.data_ptr<int>(),
                event_indices_out.data_ptr<int>(),
                t_steps * batch_size,
                n_neuron,
                compact_grid,
                kThreadsPerBlock,
                stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (overflow.item<int>() != 0) {
            TORCH_CHECK(false, "persistent SNN spike queue overflow.");
        }
    }
    return {
        dense_spikes,
        event_offsets_out,
        event_indices_out,
        v_out,
        psc_out,
        psc_h_out,
        asc_out,
        refractory_out,
        overflow,
    };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
persistent_snn_forward_dense_workspace_cuda(
    torch::Tensor event_offsets,
    torch::Tensor event_indices,
    torch::Tensor event_values,
    bool has_event_values,
    torch::Tensor graph_indptr,
    torch::Tensor graph_indices,
    torch::Tensor graph_weight,
    torch::Tensor graph_delay,
    bool has_delay,
    torch::Tensor v,
    torch::Tensor psc,
    torch::Tensor psc_h,
    torch::Tensor asc,
    torch::Tensor refractory,
    torch::Tensor dense_spikes,
    torch::Tensor v_out,
    torch::Tensor psc_out,
    torch::Tensor psc_h_out,
    torch::Tensor asc_out,
    torch::Tensor refractory_out,
    torch::Tensor input_current,
    torch::Tensor spike_queue_batch,
    torch::Tensor spike_queue_pre,
    torch::Tensor spike_count,
    torch::Tensor work_counter,
    torch::Tensor event_counts,
    torch::Tensor event_indices_full,
    torch::Tensor overflow,
    double dt,
    double tau,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double v_rest,
    double c_m,
    double tau_ref,
    double k0,
    double k1,
    double asc_amp0,
    double asc_amp1,
    double psc_g_max,
    bool hard_reset) {
    TORCH_CHECK(!hard_reset, "persistent SNN v1 only supports soft reset.");

    c10::cuda::CUDAGuard guard(v.device());
    validate_persistent_inputs(
        event_offsets,
        event_indices,
        event_values,
        has_event_values,
        graph_indptr,
        graph_indices,
        graph_weight,
        graph_delay,
        has_delay,
        v,
        psc,
        psc_h,
        asc,
        refractory,
        tau,
        tau_syn,
        c_m,
        tau_ref,
        k0,
        k1);

    const auto batch_size = static_cast<int>(v.size(0));
    const auto n_neuron = static_cast<int>(v.size(1));
    const auto t_steps =
        static_cast<int>((event_offsets.numel() - 1) / batch_size);
    TORCH_CHECK(t_steps > 0, "T must be positive.");

    check_cuda_tensor(dense_spikes, "dense_spikes", torch::kFloat32);
    check_cuda_tensor(v_out, "v_out", torch::kFloat32);
    check_cuda_tensor(psc_out, "psc_out", torch::kFloat32);
    check_cuda_tensor(psc_h_out, "psc_h_out", torch::kFloat32);
    check_cuda_tensor(asc_out, "asc_out", torch::kFloat32);
    check_cuda_tensor(refractory_out, "refractory_out", torch::kFloat32);
    check_cuda_tensor(input_current, "input_current", torch::kFloat32);
    check_cuda_tensor(spike_queue_batch, "spike_queue_batch", torch::kInt32);
    check_cuda_tensor(spike_queue_pre, "spike_queue_pre", torch::kInt32);
    check_cuda_tensor(spike_count, "spike_count", torch::kInt32);
    check_cuda_tensor(work_counter, "work_counter", torch::kInt32);
    check_cuda_tensor(event_counts, "event_counts", torch::kInt32);
    check_cuda_tensor(event_indices_full, "event_indices_full", torch::kInt32);
    check_cuda_tensor(overflow, "overflow", torch::kInt32);
    check_same_device(dense_spikes, v, "dense_spikes");
    check_same_device(v_out, v, "v_out");
    check_same_device(psc_out, v, "psc_out");
    check_same_device(psc_h_out, v, "psc_h_out");
    check_same_device(asc_out, v, "asc_out");
    check_same_device(refractory_out, v, "refractory_out");
    check_same_device(input_current, v, "input_current");
    check_same_device(spike_queue_batch, event_offsets, "spike_queue_batch");
    check_same_device(spike_queue_pre, event_offsets, "spike_queue_pre");
    check_same_device(spike_count, event_offsets, "spike_count");
    check_same_device(work_counter, event_offsets, "work_counter");
    check_same_device(event_counts, event_offsets, "event_counts");
    check_same_device(event_indices_full, event_offsets, "event_indices_full");
    check_same_device(overflow, event_offsets, "overflow");
    check_shape(v_out, v, "v_out");
    check_shape(psc_out, psc, "psc_out");
    check_shape(psc_h_out, psc_h, "psc_h_out");
    check_shape(asc_out, asc, "asc_out");
    check_shape(refractory_out, refractory, "refractory_out");
    check_shape(input_current, v, "input_current");
    TORCH_CHECK(
        dense_spikes.sizes() == torch::IntArrayRef({t_steps, batch_size, n_neuron}),
        "dense_spikes must have shape (T, B, N).");
    TORCH_CHECK(
        spike_queue_batch.numel() >= batch_size * n_neuron &&
            spike_queue_pre.numel() >= batch_size * n_neuron,
        "spike queues must have at least B * N elements.");
    TORCH_CHECK(
        spike_count.numel() == 1 && work_counter.numel() == 1 &&
            overflow.numel() == 1,
        "spike_count, work_counter, and overflow must have one element.");

    v_out.copy_(v);
    psc_out.copy_(psc);
    psc_h_out.copy_(psc_h);
    asc_out.copy_(asc);
    refractory_out.copy_(refractory);
    overflow.zero_();
    event_counts.zero_();

    const int grid_dim = cooperative_grid_dim(kThreadsPerBlock);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    launch_persistent_snn_kernel(
        event_offsets.data_ptr<int>(),
        event_indices.data_ptr<int>(),
        has_event_values ? event_values.data_ptr<float>() : nullptr,
        has_event_values,
        graph_indptr.data_ptr<int>(),
        graph_indices.data_ptr<int>(),
        graph_weight.data_ptr<float>(),
        v_out.data_ptr<float>(),
        psc_out.data_ptr<float>(),
        psc_h_out.data_ptr<float>(),
        asc_out.data_ptr<float>(),
        refractory_out.data_ptr<float>(),
        dense_spikes.data_ptr<float>(),
        input_current.data_ptr<float>(),
        spike_queue_batch.data_ptr<int>(),
        spike_queue_pre.data_ptr<int>(),
        spike_count.data_ptr<int>(),
        work_counter.data_ptr<int>(),
        event_counts.data_ptr<int>(),
        event_indices_full.data_ptr<int>(),
        overflow.data_ptr<int>(),
        false,
        t_steps,
        batch_size,
        n_neuron,
        static_cast<float>(dt),
        static_cast<float>(tau),
        static_cast<float>(tau_syn),
        static_cast<float>(v_threshold),
        static_cast<float>(v_reset),
        static_cast<float>(v_rest),
        static_cast<float>(c_m),
        static_cast<float>(tau_ref),
        static_cast<float>(k0),
        static_cast<float>(k1),
        static_cast<float>(asc_amp0),
        static_cast<float>(asc_amp1),
        static_cast<float>(psc_g_max),
        grid_dim,
        kThreadsPerBlock,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        dense_spikes,
        v_out,
        psc_out,
        psc_h_out,
        asc_out,
        refractory_out,
        overflow,
    };
}

TORCH_LIBRARY(btorch_cuda, m) {
    m.def(
        "persistent_snn_forward("
        "Tensor event_offsets, Tensor event_indices, Tensor event_values, "
        "bool has_event_values, Tensor graph_indptr, Tensor graph_indices, "
        "Tensor graph_weight, Tensor graph_delay, bool has_delay, Tensor v, "
        "Tensor psc, Tensor psc_h, Tensor asc, Tensor refractory, float dt, "
        "float tau, float tau_syn, float v_threshold, float v_reset, "
        "float v_rest, float c_m, float tau_ref, float k0, float k1, "
        "float asc_amp0, float asc_amp1, float psc_g_max, bool hard_reset, "
        "bool return_events) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "persistent_snn_forward_dense_workspace("
        "Tensor event_offsets, Tensor event_indices, Tensor event_values, "
        "bool has_event_values, Tensor graph_indptr, Tensor graph_indices, "
        "Tensor graph_weight, Tensor graph_delay, bool has_delay, Tensor v, "
        "Tensor psc, Tensor psc_h, Tensor asc, Tensor refractory, "
        "Tensor dense_spikes, Tensor v_out, Tensor psc_out, Tensor psc_h_out, "
        "Tensor asc_out, Tensor refractory_out, Tensor input_current, "
        "Tensor spike_queue_batch, Tensor spike_queue_pre, Tensor spike_count, "
        "Tensor work_counter, Tensor event_counts, Tensor event_indices_full, "
        "Tensor overflow, float dt, float tau, float tau_syn, "
        "float v_threshold, float v_reset, float v_rest, float c_m, "
        "float tau_ref, float k0, float k1, float asc_amp0, float asc_amp1, "
        "float psc_g_max, bool hard_reset) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(btorch_cuda, CUDA, m) {
    m.impl("persistent_snn_forward", &persistent_snn_forward_cuda);
    m.impl(
        "persistent_snn_forward_dense_workspace",
        &persistent_snn_forward_dense_workspace_cuda);
}
