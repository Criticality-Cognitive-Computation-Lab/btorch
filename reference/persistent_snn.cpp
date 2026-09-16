#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef BTORCH_WARP_SPEC_MODE
#define BTORCH_WARP_SPEC_MODE 0
#endif

#ifndef BTORCH_LONG_WARP_SPEC_ENABLED
#define BTORCH_LONG_WARP_SPEC_ENABLED 0
#endif

constexpr int kThreadsPerBlock = 256;
constexpr int kMinimumEdgesPerTask = 128;
constexpr int kComponentEventCount = 7;
constexpr int kComponentTimingColumns = 6;

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
#ifdef BTORCH_PERSISTENT_PIPELINE
    float* recurrent_delta_0,
    float* recurrent_delta_1,
#endif
    float* dense_spikes,
    float* input_current,
#ifdef BTORCH_PERSISTENT_PIPELINE
    int* spike_queue_neuron,
    uint32_t* spike_queue_state,
    int* queue_tail,
    int* next_ticket,
    int* update_done_blocks,
    unsigned long long* debug_counters,
    unsigned long long* timing_stats,
    int* timing_counters,
#else
    int* spike_queue_batch,
    int* spike_queue_edge_start,
    int* spike_queue_edge_end,
    int* spike_count,
    int* work_counter,
    unsigned long long* timing_stats,
#endif
    int* event_counts,
    int* event_indices_full,
    bool return_dense,
    bool return_events,
    int t_steps,
#ifndef BTORCH_PERSISTENT_PIPELINE
    int batch_size,
#endif
    int n_neuron,
#ifdef BTORCH_PERSISTENT_PIPELINE
    int queue_capacity,
    int update_block_count,
    int dedicated_consumer_warps,
    int helper_consumer_warps,
    int ticket_chunk,
    int static_waves,
    int high_fanout_threshold,
    uint32_t forward_epoch_base,
#else
    int queue_capacity,
#endif
    float dt,
    float tau_mem,
    float tau_syn,
    float v_threshold,
    float v_reset,
    float c_m,
    int grid_dim,
    int block_dim,
    cudaStream_t stream);

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
    cudaStream_t stream);

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

#ifdef BTORCH_PERSISTENT_PIPELINE
int persistent_snn_max_active_blocks_per_sm(
    int block_dim,
    bool return_dense,
    bool return_events,
    size_t dynamic_shared_bytes);
#else
int persistent_snn_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events);
#endif
int persistent_snn_binned_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events);
int persistent_snn_spike_block_max_active_blocks_per_sm(
    int block_dim, bool return_dense, bool return_events);

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

int cooperative_grid_dim_uncached(
    int block_dim, bool return_dense, bool return_events) {
    int device = -1;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties failed");
    TORCH_CHECK(
        prop.cooperativeLaunch,
        "persistent SNN requires CUDA cooperative launch support.");

#ifdef BTORCH_PERSISTENT_PIPELINE
    const int active_blocks = persistent_snn_max_active_blocks_per_sm(
        block_dim, return_dense, return_events, 0);
#else
    const int active_blocks = persistent_snn_max_active_blocks_per_sm(
        block_dim, return_dense, return_events);
#endif
    TORCH_CHECK(active_blocks > 0, "persistent SNN kernel has zero occupancy.");
    return active_blocks * prop.multiProcessorCount;
}

// `cudaGetDeviceProperties` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
// are synchronous, CPU-blocking driver calls -- and this function was
// re-running them on *every single* forward() call. Measured cost: ~1.1-1.2ms
// per call (dwarfing every `.item()` sync in this file combined, which are
// each ~10-30us). The result only depends on {block_dim, current device,
// this kernel's fixed resource usage (registers/shared mem)}, none of which
// change between calls in this process, so it's safe to compute once and
// cache. (Not safe across a device change mid-process, but this op is always
// invoked with the tensors' own device via CUDAGuard, and does not support
// multi-device dispatch within one call.)
int cooperative_grid_dim(
    int block_dim, bool return_dense, bool return_events) {
    static int cached[3] = {-1, -1, -1};
    static int cached_block_dim[3] = {-1, -1, -1};
    const int mode = return_events ? (return_dense ? 2 : 1) : 0;
    if (cached[mode] < 0 || cached_block_dim[mode] != block_dim) {
        cached[mode] = cooperative_grid_dim_uncached(
            block_dim, return_dense, return_events);
        cached_block_dim[mode] = block_dim;
    }
    return cached[mode];
}

int cooperative_grid_dim_binned(
    int block_dim, bool return_dense, bool return_events) {
    static int cached[3] = {-1, -1, -1};
    static int cached_block_dim[3] = {-1, -1, -1};
    const int mode = return_events ? (return_dense ? 2 : 1) : 0;
    if (cached[mode] < 0 || cached_block_dim[mode] != block_dim) {
        int device = -1;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");

        cudaDeviceProp prop{};
        check_cuda(
            cudaGetDeviceProperties(&prop, device),
            "cudaGetDeviceProperties failed");
        TORCH_CHECK(
            prop.cooperativeLaunch,
            "persistent SNN requires CUDA cooperative launch support.");

        const int active_blocks =
            persistent_snn_binned_max_active_blocks_per_sm(
                block_dim, return_dense, return_events);
        TORCH_CHECK(
            active_blocks > 0,
            "persistent SNN binned kernel has zero occupancy.");
        cached[mode] = active_blocks * prop.multiProcessorCount;
        cached_block_dim[mode] = block_dim;
    }
    return cached[mode];
}

int cooperative_grid_dim_spike_block(
    int block_dim, bool return_dense, bool return_events) {
    static int cached[3] = {-1, -1, -1};
    static int cached_block_dim[3] = {-1, -1, -1};
    const int mode = return_events ? (return_dense ? 2 : 1) : 0;
    if (cached[mode] < 0 || cached_block_dim[mode] != block_dim) {
        int device = -1;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");

        cudaDeviceProp prop{};
        check_cuda(
            cudaGetDeviceProperties(&prop, device),
            "cudaGetDeviceProperties failed");
        TORCH_CHECK(
            prop.cooperativeLaunch,
            "persistent SNN requires CUDA cooperative launch support.");

        const int active_blocks =
            persistent_snn_spike_block_max_active_blocks_per_sm(
                block_dim, return_dense, return_events);
        TORCH_CHECK(
            active_blocks > 0,
            "persistent SNN spike-block kernel has zero occupancy.");
        cached[mode] = active_blocks * prop.multiProcessorCount;
        cached_block_dim[mode] = block_dim;
    }
    return cached[mode];
}

int requested_cooperative_grid_dim(int maximum_grid_dim) {
    const char* value = std::getenv("BTORCH_PERSISTENT_GRID_BLOCKS");
    if (value == nullptr || value[0] == '\0') {
        return maximum_grid_dim;
    }

    errno = 0;
    char* end = nullptr;
    const long requested = std::strtol(value, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0',
        "BTORCH_PERSISTENT_GRID_BLOCKS must be a positive integer, got '",
        value,
        "'.");
    TORCH_CHECK(
        requested > 0 && requested <= maximum_grid_dim,
        "BTORCH_PERSISTENT_GRID_BLOCKS must be in [1, ",
        maximum_grid_dim,
        "], got ",
        requested,
        ".");
    return static_cast<int>(requested);
}

bool pipeline_component_timing_enabled() {
    const char* value = std::getenv("BTORCH_PIPELINE_COMPONENT_TIMING");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
        return false;
    }
    TORCH_CHECK(
        std::strcmp(value, "1") == 0,
        "BTORCH_PIPELINE_COMPONENT_TIMING must be 0 or 1, got '",
        value,
        "'.");
    return true;
}

#ifdef BTORCH_PERSISTENT_PIPELINE
int pipeline_update_block_count(int grid_dim) {
    const char* value = std::getenv("BTORCH_PIPELINE_ROLE_RATIO");
    if (value == nullptr || value[0] == '\0') {
        value = "7:1";
    }

    TORCH_CHECK(
        std::strcmp(value, "7:1") == 0 ||
            std::strcmp(value, "3:1") == 0 ||
            std::strcmp(value, "2:1") == 0 ||
            std::strcmp(value, "1:1") == 0,
        "BTORCH_PIPELINE_ROLE_RATIO must be 7:1, 3:1, 2:1, or 1:1, got '",
        value,
        "'.");
    int update_blocks = 0;
    if (std::strcmp(value, "7:1") == 0) {
        update_blocks = (grid_dim * 7) / 8;
    } else if (std::strcmp(value, "3:1") == 0) {
        update_blocks = (grid_dim * 3) / 4;
    } else if (std::strcmp(value, "2:1") == 0) {
        update_blocks = (grid_dim * 2) / 3;
    } else if (std::strcmp(value, "1:1") == 0) {
        update_blocks = grid_dim / 2;
    }
    return std::min(grid_dim - 1, std::max(1, update_blocks));
}

size_t pipeline_update_shared_memory_limit() {
    const char* value = std::getenv("BTORCH_PIPELINE_UPDATE_SMEM_KB");
    if (value == nullptr || value[0] == '\0') {
        return 32 * 1024;
    }

    errno = 0;
    char* end = nullptr;
    const long kibibytes = std::strtol(value, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0' && kibibytes > 0,
        "BTORCH_PIPELINE_UPDATE_SMEM_KB must be a positive integer, got '",
        value,
        "'.");
    TORCH_CHECK(
        static_cast<unsigned long>(kibibytes) <=
            std::numeric_limits<size_t>::max() / 1024,
        "BTORCH_PIPELINE_UPDATE_SMEM_KB is too large.");
    return static_cast<size_t>(kibibytes) * 1024;
}

int pipeline_cooperative_grid_dim(
    int maximum_grid_dim,
    int block_dim,
    bool return_dense,
    bool return_events,
    int n_neuron) {
    constexpr size_t kUpdateBytesPerNeuron = 3 * sizeof(float);
    int device = -1;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice failed");

    cudaDeviceProp prop{};
    check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties failed");
    const size_t configured_limit = pipeline_update_shared_memory_limit();
    const size_t device_limit = static_cast<size_t>(
        prop.sharedMemPerBlockOptin > 0 ? prop.sharedMemPerBlockOptin
                                       : prop.sharedMemPerBlock);
    const size_t shared_limit = std::min(configured_limit, device_limit);

    const char* requested_value =
        std::getenv("BTORCH_PERSISTENT_GRID_BLOCKS");
    const bool has_requested_grid =
        requested_value != nullptr && requested_value[0] != '\0';
    const int requested_grid =
        requested_cooperative_grid_dim(maximum_grid_dim);

    auto configuration_fits = [&](int candidate_grid) {
        if (candidate_grid < 2) {
            return false;
        }
        const int update_blocks =
            pipeline_update_block_count(candidate_grid);
        const int neurons_per_block =
            (n_neuron + update_blocks - 1) / update_blocks;
        const size_t shared_bytes =
            static_cast<size_t>(neurons_per_block) *
            kUpdateBytesPerNeuron;
        if (shared_bytes > shared_limit) {
            return false;
        }
        const int active_blocks = persistent_snn_max_active_blocks_per_sm(
            block_dim,
            return_dense,
            return_events,
            shared_bytes);
        return active_blocks > 0 &&
            candidate_grid <= active_blocks * prop.multiProcessorCount;
    };

    if (has_requested_grid) {
        TORCH_CHECK(
            configuration_fits(requested_grid),
            "BTORCH_PERSISTENT_GRID_BLOCKS=",
            requested_grid,
            " cannot stage all ",
            n_neuron,
            " neurons within the ",
            configured_limit / 1024,
            " KiB UPDATE shared-memory budget and cooperative occupancy.");
        return requested_grid;
    }

    const int maximum_blocks_per_sm =
        maximum_grid_dim / prop.multiProcessorCount;
    for (int blocks_per_sm = maximum_blocks_per_sm; blocks_per_sm >= 1;
         --blocks_per_sm) {
        const int candidate_grid =
            blocks_per_sm * prop.multiProcessorCount;
        if (configuration_fits(candidate_grid)) {
            return candidate_grid;
        }
    }
    TORCH_CHECK(
        false,
        "No cooperative pipeline grid can stage all ",
        n_neuron,
        " neurons within the ",
        configured_limit / 1024,
        " KiB UPDATE shared-memory budget (device limit ",
        device_limit / 1024,
        " KiB).");
    return 0;
}

bool pipeline_debug_counters_enabled() {
    const char* value = std::getenv("BTORCH_PIPELINE_DEBUG_COUNTERS");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
        return false;
    }
    TORCH_CHECK(
        std::strcmp(value, "1") == 0,
        "BTORCH_PIPELINE_DEBUG_COUNTERS must be 0 or 1, got '",
        value,
        "'.");
    return true;
}

int pipeline_consumer_warps(
    const char* environment_name, int default_warps) {
    const char* value = std::getenv(environment_name);
    if (value == nullptr || value[0] == '\0') {
        return default_warps;
    }
    errno = 0;
    char* end = nullptr;
    const long warps = std::strtol(value, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0' &&
            (warps == 1 || warps == 2 || warps == 4 || warps == 8),
        environment_name,
        " must be 1, 2, 4, or 8, got '",
        value,
        "'.");
    return static_cast<int>(warps);
}

int pipeline_dedicated_consumer_warps() {
    return pipeline_consumer_warps(
        "BTORCH_PIPELINE_DEDICATED_WARPS", 1);
}

int pipeline_helper_consumer_warps() {
    return pipeline_consumer_warps(
        "BTORCH_PIPELINE_HELPER_WARPS", 1);
}

int pipeline_ticket_chunk() {
    const char* value = std::getenv("BTORCH_PIPELINE_TICKET_CHUNK");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    errno = 0;
    char* end = nullptr;
    const long chunk = std::strtol(value, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0' &&
            (chunk == 1 || chunk == 2 || chunk == 4),
        "BTORCH_PIPELINE_TICKET_CHUNK must be 1, 2, or 4, got '",
        value,
        "'.");
    return static_cast<int>(chunk);
}

int pipeline_static_waves() {
    const char* value = std::getenv("BTORCH_PIPELINE_STATIC_WAVES");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    const long waves = std::strtol(value, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0' &&
            (waves == 0 || waves == 1 || waves == 2 || waves == 4),
        "BTORCH_PIPELINE_STATIC_WAVES must be 0, 1, 2, or 4, got '",
        value,
        "'.");
    return static_cast<int>(waves);
}

int pipeline_binned_threshold() {
    const char* value = std::getenv("BTORCH_PIPELINE_BINNED_THRESHOLD");
    if (value == nullptr || value[0] == '\0') {
        return 512;
    }
    errno = 0;
    char* end = nullptr;
    const long threshold = std::strtol(value, &end, 10);
    TORCH_CHECK(
        errno == 0 && end != value && *end == '\0' &&
            (threshold == 0 || threshold == 128 || threshold == 256 ||
             threshold == 512),
        "BTORCH_PIPELINE_BINNED_THRESHOLD must be 0, 128, 256, or 512, got '",
        value,
        "'.");
    return static_cast<int>(threshold);
}

std::tuple<uint32_t, bool> pipeline_forward_epoch(
    const void* queue_state, int t_steps) {
    constexpr uint64_t kEpochCount = 0xffffu;
    static std::mutex epoch_mutex;
    static std::unordered_map<const void*, uint64_t> next_epoch;
    std::lock_guard<std::mutex> lock(epoch_mutex);

    uint64_t& sequence = next_epoch[queue_state];
    const uint32_t epoch_base =
        static_cast<uint32_t>(sequence % kEpochCount);
    const bool must_clear =
        sequence == 0 || epoch_base == 0 ||
        epoch_base + t_steps > kEpochCount;
    sequence += static_cast<uint64_t>(t_steps);
    return {epoch_base, must_clear};
}
#endif

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
persistent_snn_forward_cuda_impl(
    torch::Tensor event_offsets,
    torch::Tensor event_indices,
    torch::Tensor event_values,
    bool has_event_values,
    torch::Tensor graph_indptr,
    torch::Tensor graph_indices,
    torch::Tensor graph_weight,
    bool graph_indices_validated,
    torch::Tensor input_current,
    torch::Tensor spike_queue_batch,
    torch::Tensor spike_queue_edge_start,
    torch::Tensor spike_queue_edge_end,
    torch::Tensor spike_count,
    torch::Tensor work_counter,
    torch::Tensor graph_delay,
    bool has_delay,
    bool delay_validated,
    torch::Tensor v,
    torch::Tensor psc,
    double dt,
    double tau_mem,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double c_m,
    bool hard_reset,
    bool return_dense,
    bool return_events,
    torch::Tensor graph_high_fanout,
    bool fanout_binning,
    bool spike_block,
    torch::Tensor pipeline_delta_0,
    torch::Tensor pipeline_delta_1,
    bool pipeline_preallocated_delta,
    bool pipeline_fold_psc) {
    TORCH_CHECK(
        !(fanout_binning && spike_block),
        "fanout_binning and spike_block are mutually exclusive.");
    TORCH_CHECK(!hard_reset, "persistent SNN v1 only supports soft reset.");
    check_cuda_tensor(event_offsets, "event_offsets", torch::kInt32);
    check_cuda_tensor(event_indices, "event_indices", torch::kInt32);
    check_cuda_tensor(graph_indptr, "graph_indptr", torch::kInt32);
    check_cuda_tensor(graph_indices, "graph_indices", torch::kInt32);
    check_cuda_tensor(graph_weight, "graph_weight", torch::kFloat32);
    check_cuda_tensor(input_current, "input_current", torch::kFloat32);
    check_cuda_tensor(spike_queue_batch, "spike_queue_batch", torch::kInt32);
    check_cuda_tensor(
        spike_queue_edge_start, "spike_queue_edge_start", torch::kInt32);
    check_cuda_tensor(spike_queue_edge_end, "spike_queue_edge_end", torch::kInt32);
    check_cuda_tensor(spike_count, "spike_count", torch::kInt32);
    check_cuda_tensor(work_counter, "work_counter", torch::kInt32);
    check_cuda_tensor(v, "v", torch::kFloat32);
    check_cuda_tensor(psc, "psc", torch::kFloat32);
    check_same_device(event_offsets, v, "event_offsets");
    check_same_device(event_indices, v, "event_indices");
    check_same_device(graph_indptr, v, "graph_indptr");
    check_same_device(graph_indices, v, "graph_indices");
    check_same_device(graph_weight, v, "graph_weight");
    check_same_device(input_current, v, "input_current");
    check_same_device(spike_queue_batch, v, "spike_queue_batch");
    check_same_device(spike_queue_edge_start, v, "spike_queue_edge_start");
    check_same_device(spike_queue_edge_end, v, "spike_queue_edge_end");
    check_same_device(spike_count, v, "spike_count");
    check_same_device(work_counter, v, "work_counter");
    check_same_device(psc, v, "psc");
    if (pipeline_preallocated_delta) {
        check_cuda_tensor(
            pipeline_delta_0, "pipeline_delta_0", torch::kFloat32);
        check_cuda_tensor(
            pipeline_delta_1, "pipeline_delta_1", torch::kFloat32);
        check_same_device(pipeline_delta_0, v, "pipeline_delta_0");
        check_same_device(pipeline_delta_1, v, "pipeline_delta_1");
        TORCH_CHECK(
            pipeline_delta_0.sizes() == psc.sizes() &&
                pipeline_delta_1.sizes() == psc.sizes(),
            "preallocated pipeline deltas must match psc shape.");
    }
    if (fanout_binning) {
        check_cuda_tensor(
            graph_high_fanout, "graph_high_fanout", torch::kInt32);
        check_same_device(graph_high_fanout, v, "graph_high_fanout");
    }
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

    c10::cuda::CUDAGuard guard(v.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool component_timing = pipeline_component_timing_enabled();
#ifdef ENABLE_PIPELINE_TIMING
    TORCH_CHECK(
        !component_timing,
        "pipeline overlap timing and component timing are mutually exclusive.");
#endif
    TORCH_CHECK(
        !component_timing || (!fanout_binning && !spike_block),
        "component timing requires the plain persistent provider.");
    std::array<cudaEvent_t, kComponentEventCount> component_events{};
    if (component_timing) {
        for (cudaEvent_t& event : component_events) {
            const cudaError_t error = cudaEventCreate(&event);
            TORCH_CHECK(
                error == cudaSuccess,
                "failed to create component timing event: ",
                cudaGetErrorString(error));
        }
        cudaEventRecord(component_events[0], stream);
    }
    if (has_delay && !delay_validated) {
        // Only pay for this O(E) reduction + device->host sync when the caller
        // (the Python `persistent_snn_forward` dispatcher) hasn't already
        // verified it. The dispatcher caches this per delay-tensor identity,
        // since the delay array is part of a graph's fixed structure and is
        // typically reused unchanged across many forward() calls. Direct
        // callers of this op (bypassing the dispatcher) always re-verify here.
        TORCH_CHECK(
            graph_delay.eq(0).all().item<bool>(),
            "persistent SNN v1 does not support nonzero delay.");
    }

    TORCH_CHECK(v.dim() == 2, "v must have shape (B, N).");
    TORCH_CHECK(psc.sizes() == v.sizes(), "psc must match v shape.");
    const auto batch_size = static_cast<int>(v.size(0));
    const auto n_neuron = static_cast<int>(v.size(1));
    TORCH_CHECK(batch_size > 0 && n_neuron > 0, "B and N must be positive.");
#ifdef BTORCH_PERSISTENT_PIPELINE
    if (!fanout_binning && !spike_block) {
        TORCH_CHECK(
            batch_size == 1,
            "persistent SNN pipeline requires batch_size == 1.");
    }
#endif
    TORCH_CHECK(
        graph_indptr.numel() == n_neuron + 1,
        "graph_indptr must have shape (N + 1,).");
    TORCH_CHECK(
        graph_indices.numel() == graph_weight.numel(),
        "graph_indices and graph_weight must match.");
    if (fanout_binning) {
        TORCH_CHECK(
            graph_high_fanout.dim() == 1 &&
                graph_high_fanout.numel() == n_neuron,
            "graph_high_fanout must have shape (N,).");
    }
    if (!graph_indices_validated) {
        TORCH_CHECK(
            graph_indices.ge(0).logical_and(
                graph_indices.lt(n_neuron)).all().item<bool>(),
            "graph_indices must be in the range [0, N).");
    }
    TORCH_CHECK(event_offsets.dim() == 1, "event_offsets must be 1D.");
    TORCH_CHECK(event_indices.dim() == 1, "event_indices must be 1D.");
    TORCH_CHECK(
        event_offsets.numel() >= 2,
        "event_offsets must contain at least one bucket.");
    // event_offsets[-1] == event_indices.numel() is guaranteed by the Python
    // `persistent_snn_forward` dispatcher's `_validate_events` (which already
    // pays this device->host sync once). Not re-checked here to avoid paying
    // it a second time on every call; direct callers of this op bypass that
    // guarantee.
    TORCH_CHECK(
        (event_offsets.numel() - 1) % batch_size == 0,
        "event_offsets bucket count must be divisible by batch size.");
    const auto t_steps =
        static_cast<int>((event_offsets.numel() - 1) / batch_size);
    TORCH_CHECK(t_steps > 0, "T must be positive.");
#ifdef BTORCH_PERSISTENT_PIPELINE
    if (!fanout_binning && !spike_block) {
        TORCH_CHECK(
            t_steps < (1 << 16),
            "persistent SNN pipeline supports fewer than 2^16 timesteps.");
    }
#endif
    TORCH_CHECK(
        (n_neuron + 31) / 32 < (1 << 24),
        "spike-block descriptor supports fewer than 2^24 neuron blocks.");
    TORCH_CHECK(
        tau_mem > 0.0 && tau_syn > 0.0 && c_m > 0.0,
        "tau_mem, tau_syn, and c_m must be positive.");

    const auto options_f = v.options();
    const auto options_i = event_offsets.options();
    auto v_out = v.clone();
    auto psc_out = psc.clone();
    if (component_timing) {
        cudaEventRecord(component_events[1], stream);
    }
    auto dense_spikes = return_dense
        ? torch::empty({t_steps, batch_size, n_neuron}, options_f)
        : torch::empty({0}, options_f);
    const int64_t edge_count = graph_indices.numel();
    const int64_t tasks_per_batch = spike_block
        ? ((n_neuron + 31) / 32 + n_neuron +
           (edge_count + kMinimumEdgesPerTask - 1) /
               kMinimumEdgesPerTask)
        : (n_neuron + (edge_count + kMinimumEdgesPerTask - 1) /
                          kMinimumEdgesPerTask);
    const int64_t queue_capacity_64 =
        static_cast<int64_t>(batch_size) * tasks_per_batch;
    TORCH_CHECK(
        queue_capacity_64 <= std::numeric_limits<int>::max(),
        "persistent SNN work queue is too large for int32 indexing.");
    const auto queue_capacity = static_cast<int>(queue_capacity_64);
    TORCH_CHECK(
        input_current.numel() == batch_size * n_neuron,
        "input_current must have shape (B, N).");
    TORCH_CHECK(
        spike_queue_batch.numel() >= queue_capacity &&
            spike_queue_edge_start.numel() >= queue_capacity &&
            spike_queue_edge_end.numel() >= queue_capacity,
        "persistent SNN task queues are too small.");
    const int counter_size =
#ifdef BTORCH_PERSISTENT_PIPELINE
        (!fanout_binning && !spike_block) ? 2 :
#endif
        ((fanout_binning || spike_block) ? 2 : 1);
    TORCH_CHECK(
        spike_count.numel() >= counter_size,
        "spike_count does not have enough counters.");
    TORCH_CHECK(
        work_counter.numel() >= counter_size,
        "work_counter does not have enough counters.");
    auto event_counts = return_events
        ? torch::zeros({t_steps * batch_size}, options_i)
        : torch::empty({0}, options_i);
    auto event_indices_full = return_events
        ? torch::empty({t_steps * batch_size * n_neuron}, options_i)
        : torch::empty({0}, options_i);
#ifdef ENABLE_BLOCK_STATS
    constexpr int kBlockStatsColumns =
        BTORCH_LONG_WARP_SPEC_ENABLED
        ? 64
        : (BTORCH_WARP_SPEC_MODE == 4 ? 44 : 23);
    const int64_t block_stats_records =
        static_cast<int64_t>(t_steps) * batch_size * ((n_neuron + 31) / 32);
    auto overflow = spike_block && return_dense
        ? torch::zeros(
              {block_stats_records, kBlockStatsColumns}, options_i)
        : torch::empty({0}, options_i);
#else
    auto overflow = torch::empty({0}, options_i);
#endif
#ifdef ENABLE_PIPELINE_TIMING
#ifdef BTORCH_PERSISTENT_PIPELINE
    constexpr int kPipelineTimingStatsColumns = 9;
#else
    constexpr int kPipelineTimingStatsColumns = 5;
#endif
    auto pipeline_timing_stats =
        torch::zeros(
            {t_steps, kPipelineTimingStatsColumns},
            options_i.dtype(torch::kInt64));
    if (!fanout_binning && !spike_block) {
        overflow = pipeline_timing_stats;
    }
#else
    auto pipeline_timing_stats = torch::empty({0}, options_i);
#endif
#ifdef BTORCH_PERSISTENT_PIPELINE
    const bool pipeline_debug_counters =
        !fanout_binning && !spike_block &&
        pipeline_debug_counters_enabled();
#ifdef ENABLE_PIPELINE_TIMING
    TORCH_CHECK(
        !pipeline_debug_counters,
        "pipeline debug counters and timing cannot be enabled together.");
    auto pipeline_timing_counters = torch::zeros({3}, options_i);
#else
    auto pipeline_timing_counters = torch::empty({0}, options_i);
#endif
    if (pipeline_debug_counters) {
        overflow = torch::zeros({14}, options_i.dtype(torch::kInt64));
    }
#endif

    const int maximum_grid_dim = spike_block
        ? cooperative_grid_dim_spike_block(
              kThreadsPerBlock, return_dense, return_events)
        : (fanout_binning
               ? cooperative_grid_dim_binned(
                     kThreadsPerBlock, return_dense, return_events)
               : cooperative_grid_dim(
                     kThreadsPerBlock, return_dense, return_events));
#ifdef BTORCH_PERSISTENT_PIPELINE
    const int grid_dim = !spike_block && !fanout_binning
        ? pipeline_cooperative_grid_dim(
              maximum_grid_dim,
              kThreadsPerBlock,
              return_dense,
              return_events,
              n_neuron)
        : requested_cooperative_grid_dim(maximum_grid_dim);
#else
    const int grid_dim =
        requested_cooperative_grid_dim(maximum_grid_dim);
#endif
    if (component_timing) {
        cudaEventRecord(component_events[2], stream);
    }
    if (spike_block) {
        launch_persistent_snn_spike_block_kernel(
            event_offsets.data_ptr<int>(),
            event_indices.data_ptr<int>(),
            has_event_values ? event_values.data_ptr<float>() : nullptr,
            has_event_values,
            graph_indptr.data_ptr<int>(),
            graph_indices.data_ptr<int>(),
            graph_weight.data_ptr<float>(),
            v_out.data_ptr<float>(),
            psc_out.data_ptr<float>(),
            return_dense ? dense_spikes.data_ptr<float>() : nullptr,
            input_current.data_ptr<float>(),
            spike_queue_batch.data_ptr<int>(),
            spike_queue_edge_start.data_ptr<int>(),
            spike_queue_edge_end.data_ptr<int>(),
            spike_count.data_ptr<int>(),
            work_counter.data_ptr<int>(),
            return_events ? event_counts.data_ptr<int>() : nullptr,
            return_events ? event_indices_full.data_ptr<int>() : nullptr,
            overflow.numel() ? overflow.data_ptr<int>() : nullptr,
            return_dense,
            return_events,
            t_steps,
            batch_size,
            n_neuron,
            queue_capacity,
            static_cast<float>(dt),
            static_cast<float>(tau_mem),
            static_cast<float>(tau_syn),
            static_cast<float>(v_threshold),
            static_cast<float>(v_reset),
            static_cast<float>(c_m),
            grid_dim,
            kThreadsPerBlock,
            stream);
    } else if (fanout_binning) {
        launch_persistent_snn_binned_kernel(
            event_offsets.data_ptr<int>(),
            event_indices.data_ptr<int>(),
            has_event_values ? event_values.data_ptr<float>() : nullptr,
            has_event_values,
            graph_indptr.data_ptr<int>(),
            graph_indices.data_ptr<int>(),
            graph_weight.data_ptr<float>(),
            graph_high_fanout.data_ptr<int>(),
            v_out.data_ptr<float>(),
            psc_out.data_ptr<float>(),
            return_dense ? dense_spikes.data_ptr<float>() : nullptr,
            input_current.data_ptr<float>(),
            spike_queue_batch.data_ptr<int>(),
            spike_queue_edge_start.data_ptr<int>(),
            spike_queue_edge_end.data_ptr<int>(),
            spike_count.data_ptr<int>(),
            work_counter.data_ptr<int>(),
            return_events ? event_counts.data_ptr<int>() : nullptr,
            return_events ? event_indices_full.data_ptr<int>() : nullptr,
            return_dense,
            return_events,
            t_steps,
            batch_size,
            n_neuron,
            queue_capacity,
            static_cast<float>(dt),
            static_cast<float>(tau_mem),
            static_cast<float>(tau_syn),
            static_cast<float>(v_threshold),
            static_cast<float>(v_reset),
            static_cast<float>(c_m),
            grid_dim,
            kThreadsPerBlock,
            stream);
    } else {
#ifdef BTORCH_PERSISTENT_PIPELINE
        TORCH_CHECK(
            grid_dim >= 2,
            "persistent SNN pipeline requires at least two cooperative "
            "blocks.");
        const int update_block_count =
            pipeline_update_block_count(grid_dim);
        const int dedicated_consumer_warps =
            pipeline_dedicated_consumer_warps();
        const int helper_consumer_warps =
            pipeline_helper_consumer_warps();
        const int ticket_chunk = pipeline_ticket_chunk();
        const int static_waves = pipeline_static_waves();
        const int high_fanout_threshold = pipeline_binned_threshold();
        TORCH_CHECK(
            high_fanout_threshold == 0 ||
                (ticket_chunk == 1 && static_waves == 0),
            "pipeline binning currently requires ticket chunk 1 and static "
            "waves 0.");
        const auto [forward_epoch_base, must_clear_queue_state] =
            pipeline_forward_epoch(
                spike_queue_edge_start.data_ptr<int>(), t_steps);
        auto recurrent_delta_0 = pipeline_preallocated_delta
            ? pipeline_delta_0
            : torch::zeros_like(psc_out);
        auto recurrent_delta_1 = pipeline_preallocated_delta
            ? pipeline_delta_1
            : torch::zeros_like(psc_out);
        if (pipeline_preallocated_delta) {
            recurrent_delta_0.zero_();
            recurrent_delta_1.zero_();
        }
        auto pipeline_update_done = torch::zeros({1}, options_i);
        if (component_timing) {
            cudaEventRecord(component_events[3], stream);
        }
        if (must_clear_queue_state) {
            spike_queue_edge_start.zero_();
        }
        if (component_timing) {
            cudaEventRecord(component_events[4], stream);
        }
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
            recurrent_delta_0.data_ptr<float>(),
            recurrent_delta_1.data_ptr<float>(),
            return_dense ? dense_spikes.data_ptr<float>() : nullptr,
            input_current.data_ptr<float>(),
            spike_queue_batch.data_ptr<int>(),
            reinterpret_cast<uint32_t*>(
                spike_queue_edge_start.data_ptr<int>()),
            spike_count.data_ptr<int>(),
            work_counter.data_ptr<int>(),
            pipeline_update_done.data_ptr<int>(),
            pipeline_debug_counters
                ? reinterpret_cast<unsigned long long*>(
                      overflow.data_ptr<int64_t>())
                : nullptr,
#ifdef ENABLE_PIPELINE_TIMING
            reinterpret_cast<unsigned long long*>(
                pipeline_timing_stats.data_ptr<int64_t>()),
            pipeline_timing_counters.data_ptr<int>(),
#else
            nullptr,
            nullptr,
#endif
            return_events ? event_counts.data_ptr<int>() : nullptr,
            return_events ? event_indices_full.data_ptr<int>() : nullptr,
            return_dense,
            return_events,
            t_steps,
            n_neuron,
            queue_capacity,
            update_block_count,
            dedicated_consumer_warps,
            helper_consumer_warps,
            ticket_chunk,
            static_waves,
            high_fanout_threshold,
            forward_epoch_base,
            static_cast<float>(dt),
            static_cast<float>(tau_mem),
            static_cast<float>(tau_syn),
            static_cast<float>(v_threshold),
            static_cast<float>(v_reset),
            static_cast<float>(c_m),
            grid_dim,
            kThreadsPerBlock,
            stream);
        if (component_timing) {
            cudaEventRecord(component_events[5], stream);
        }
        // The split PSC representation carries the last timestep's newly
        // generated recurrent current in the write delta. Fold it back into
        // psc_out so state remains compatible across separate forward calls.
        if (pipeline_fold_psc) {
            psc_out.add_(
                (t_steps & 1) ? recurrent_delta_1 : recurrent_delta_0);
        }
        if (component_timing) {
            cudaEventRecord(component_events[6], stream);
        }
#else
        TORCH_CHECK(
            !pipeline_preallocated_delta,
            "preallocated deltas require the pipeline build.");
        if (component_timing) {
            cudaEventRecord(component_events[3], stream);
            cudaEventRecord(component_events[4], stream);
        }
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
            return_dense ? dense_spikes.data_ptr<float>() : nullptr,
            input_current.data_ptr<float>(),
            spike_queue_batch.data_ptr<int>(),
            spike_queue_edge_start.data_ptr<int>(),
            spike_queue_edge_end.data_ptr<int>(),
            spike_count.data_ptr<int>(),
            work_counter.data_ptr<int>(),
#ifdef ENABLE_PIPELINE_TIMING
            reinterpret_cast<unsigned long long*>(
                pipeline_timing_stats.data_ptr<int64_t>()),
#else
            nullptr,
#endif
            return_events ? event_counts.data_ptr<int>() : nullptr,
            return_events ? event_indices_full.data_ptr<int>() : nullptr,
            return_dense,
            return_events,
            t_steps,
            batch_size,
            n_neuron,
            queue_capacity,
            static_cast<float>(dt),
            static_cast<float>(tau_mem),
            static_cast<float>(tau_syn),
            static_cast<float>(v_threshold),
            static_cast<float>(v_reset),
            static_cast<float>(c_m),
            grid_dim,
            kThreadsPerBlock,
            stream);
        if (component_timing) {
            cudaEventRecord(component_events[5], stream);
            cudaEventRecord(component_events[6], stream);
        }
#endif
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (component_timing) {
        const cudaError_t sync_error = cudaEventSynchronize(component_events[6]);
        TORCH_CHECK(
            sync_error == cudaSuccess,
            "failed to synchronize component timing event: ",
            cudaGetErrorString(sync_error));
        std::vector<float> elapsed(kComponentTimingColumns, 0.0f);
        cudaEventElapsedTime(
            &elapsed[0], component_events[0], component_events[1]);
        cudaEventElapsedTime(
            &elapsed[1], component_events[2], component_events[3]);
        cudaEventElapsedTime(
            &elapsed[2], component_events[3], component_events[4]);
        cudaEventElapsedTime(
            &elapsed[3], component_events[4], component_events[5]);
        cudaEventElapsedTime(
            &elapsed[4], component_events[5], component_events[6]);
        cudaEventElapsedTime(
            &elapsed[5], component_events[0], component_events[6]);
        for (cudaEvent_t event : component_events) {
            cudaEventDestroy(event);
        }
        overflow = torch::tensor(
            elapsed, torch::TensorOptions().dtype(torch::kFloat32));
    }

    torch::Tensor event_offsets_out;
    torch::Tensor event_indices_out;
    if (return_events) {
        event_offsets_out = torch::empty({t_steps * batch_size + 1}, options_i);
        event_offsets_out[0].zero_();
        event_offsets_out.slice(0, 1).copy_(torch::cumsum(event_counts, 0));
        const int total_spikes =
            event_offsets_out[event_offsets_out.numel() - 1].item<int>();
        event_indices_out = torch::empty({total_spikes}, options_i);
        if (total_spikes > 0) {
            const int compact_grid = std::min(
                grid_dim,
                (t_steps * batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
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
    } else {
        // Caller doesn't want per-spike event output (return_mode="dense") --
        // skip the cumsum + device->host size readback entirely instead of
        // paying for it unconditionally on every call.
        event_offsets_out = torch::empty({0}, options_i);
        event_indices_out = torch::empty({0}, options_i);
    }

    return {
        dense_spikes,
        event_offsets_out,
        event_indices_out,
        v_out,
        psc_out,
        overflow,
    };
}

std::tuple<
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
    bool graph_indices_validated,
    torch::Tensor input_current,
    torch::Tensor spike_queue_batch,
    torch::Tensor spike_queue_edge_start,
    torch::Tensor spike_queue_edge_end,
    torch::Tensor spike_count,
    torch::Tensor work_counter,
    torch::Tensor graph_delay,
    bool has_delay,
    bool delay_validated,
    torch::Tensor v,
    torch::Tensor psc,
    double dt,
    double tau_mem,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double c_m,
    bool hard_reset,
    bool return_dense,
    bool return_events,
    const std::optional<torch::Tensor>& pipeline_delta_0,
    const std::optional<torch::Tensor>& pipeline_delta_1,
    bool pipeline_fold_psc) {
    TORCH_CHECK(
        pipeline_delta_0.has_value() == pipeline_delta_1.has_value(),
        "pipeline_delta_0 and pipeline_delta_1 must be provided together.");
    const bool pipeline_preallocated_delta =
        pipeline_delta_0.has_value();
    return persistent_snn_forward_cuda_impl(
        event_offsets,
        event_indices,
        event_values,
        has_event_values,
        graph_indptr,
        graph_indices,
        graph_weight,
        graph_indices_validated,
        input_current,
        spike_queue_batch,
        spike_queue_edge_start,
        spike_queue_edge_end,
        spike_count,
        work_counter,
        graph_delay,
        has_delay,
        delay_validated,
        v,
        psc,
        dt,
        tau_mem,
        tau_syn,
        v_threshold,
        v_reset,
        c_m,
        hard_reset,
        return_dense,
        return_events,
        torch::Tensor(),
        false,
        false,
        pipeline_delta_0.value_or(torch::Tensor()),
        pipeline_delta_1.value_or(torch::Tensor()),
        pipeline_preallocated_delta,
        pipeline_fold_psc);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
persistent_snn_forward_binned_cuda(
    torch::Tensor event_offsets,
    torch::Tensor event_indices,
    torch::Tensor event_values,
    bool has_event_values,
    torch::Tensor graph_indptr,
    torch::Tensor graph_indices,
    torch::Tensor graph_weight,
    bool graph_indices_validated,
    torch::Tensor input_current,
    torch::Tensor spike_queue_batch,
    torch::Tensor spike_queue_edge_start,
    torch::Tensor spike_queue_edge_end,
    torch::Tensor spike_count,
    torch::Tensor work_counter,
    torch::Tensor graph_high_fanout,
    torch::Tensor graph_delay,
    bool has_delay,
    bool delay_validated,
    torch::Tensor v,
    torch::Tensor psc,
    double dt,
    double tau_mem,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double c_m,
    bool hard_reset,
    bool return_dense,
    bool return_events) {
    TORCH_CHECK(!hard_reset, "persistent SNN v1 only supports soft reset.");
    check_cuda_tensor(event_offsets, "event_offsets", torch::kInt32);
    check_cuda_tensor(event_indices, "event_indices", torch::kInt32);
    check_cuda_tensor(graph_indptr, "graph_indptr", torch::kInt32);
    check_cuda_tensor(graph_indices, "graph_indices", torch::kInt32);
    check_cuda_tensor(graph_weight, "graph_weight", torch::kFloat32);
    check_cuda_tensor(graph_high_fanout, "graph_high_fanout", torch::kInt32);
    check_cuda_tensor(v, "v", torch::kFloat32);
    check_cuda_tensor(psc, "psc", torch::kFloat32);
    check_same_device(event_offsets, v, "event_offsets");
    check_same_device(event_indices, v, "event_indices");
    check_same_device(graph_indptr, v, "graph_indptr");
    check_same_device(graph_indices, v, "graph_indices");
    check_same_device(graph_weight, v, "graph_weight");
    check_same_device(graph_high_fanout, v, "graph_high_fanout");
    check_same_device(psc, v, "psc");
    TORCH_CHECK(v.dim() == 2, "v must have shape (B, N).");
    TORCH_CHECK(
        graph_high_fanout.dim() == 1 &&
            graph_high_fanout.numel() == v.size(1),
        "graph_high_fanout must have shape (N,).");

    return persistent_snn_forward_cuda_impl(
        event_offsets,
        event_indices,
        event_values,
        has_event_values,
        graph_indptr,
        graph_indices,
        graph_weight,
        graph_indices_validated,
        input_current,
        spike_queue_batch,
        spike_queue_edge_start,
        spike_queue_edge_end,
        spike_count,
        work_counter,
        graph_delay,
        has_delay,
        delay_validated,
        v,
        psc,
        dt,
        tau_mem,
        tau_syn,
        v_threshold,
        v_reset,
        c_m,
        hard_reset,
        return_dense,
        return_events,
        graph_high_fanout,
        true,
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        true);

}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
persistent_snn_forward_spike_block_cuda(
    torch::Tensor event_offsets,
    torch::Tensor event_indices,
    torch::Tensor event_values,
    bool has_event_values,
    torch::Tensor graph_indptr,
    torch::Tensor graph_indices,
    torch::Tensor graph_weight,
    bool graph_indices_validated,
    torch::Tensor input_current,
    torch::Tensor spike_queue_batch,
    torch::Tensor spike_queue_edge_start,
    torch::Tensor spike_queue_edge_end,
    torch::Tensor spike_count,
    torch::Tensor work_counter,
    torch::Tensor graph_delay,
    bool has_delay,
    bool delay_validated,
    torch::Tensor v,
    torch::Tensor psc,
    double dt,
    double tau_mem,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double c_m,
    bool hard_reset,
    bool return_dense,
    bool return_events) {
    return persistent_snn_forward_cuda_impl(
        event_offsets,
        event_indices,
        event_values,
        has_event_values,
        graph_indptr,
        graph_indices,
        graph_weight,
        graph_indices_validated,
        input_current,
        spike_queue_batch,
        spike_queue_edge_start,
        spike_queue_edge_end,
        spike_count,
        work_counter,
        graph_delay,
        has_delay,
        delay_validated,
        v,
        psc,
        dt,
        tau_mem,
        tau_syn,
        v_threshold,
        v_reset,
        c_m,
        hard_reset,
        return_dense,
        return_events,
        torch::Tensor(),
        false,
        true,
        torch::Tensor(),
        torch::Tensor(),
        false,
        true);
}

TORCH_LIBRARY(btorch_cuda, m) {
    m.def(
        "persistent_snn_forward("
        "Tensor event_offsets, Tensor event_indices, Tensor event_values, "
        "bool has_event_values, Tensor graph_indptr, Tensor graph_indices, "
        "Tensor graph_weight, bool graph_indices_validated, "
        "Tensor input_current, "
        "Tensor spike_queue_batch, Tensor spike_queue_edge_start, "
        "Tensor spike_queue_edge_end, Tensor spike_count, Tensor work_counter, "
        "Tensor graph_delay, bool has_delay, "
        "bool delay_validated, Tensor v, "
        "Tensor psc, float dt, float tau_mem, float tau_syn, "
        "float v_threshold, float v_reset, float c_m, bool hard_reset, "
        "bool return_dense, bool return_events, "
        "Tensor? pipeline_delta_0=None, Tensor? pipeline_delta_1=None, "
        "bool pipeline_fold_psc=True) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "persistent_snn_forward_binned("
        "Tensor event_offsets, Tensor event_indices, Tensor event_values, "
        "bool has_event_values, Tensor graph_indptr, Tensor graph_indices, "
        "Tensor graph_weight, bool graph_indices_validated, "
        "Tensor input_current, "
        "Tensor spike_queue_batch, Tensor spike_queue_edge_start, "
        "Tensor spike_queue_edge_end, Tensor spike_count, Tensor work_counter, "
        "Tensor graph_high_fanout, "
        "Tensor graph_delay, bool has_delay, "
        "bool delay_validated, Tensor v, "
        "Tensor psc, float dt, float tau_mem, float tau_syn, "
        "float v_threshold, float v_reset, float c_m, bool hard_reset, "
        "bool return_dense, bool return_events) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "persistent_snn_forward_spike_block("
        "Tensor event_offsets, Tensor event_indices, Tensor event_values, "
        "bool has_event_values, Tensor graph_indptr, Tensor graph_indices, "
        "Tensor graph_weight, bool graph_indices_validated, "
        "Tensor input_current, "
        "Tensor spike_queue_batch, Tensor spike_queue_edge_start, "
        "Tensor spike_queue_edge_end, Tensor spike_count, Tensor work_counter, "
        "Tensor graph_delay, bool has_delay, "
        "bool delay_validated, Tensor v, "
        "Tensor psc, float dt, float tau_mem, float tau_syn, "
        "float v_threshold, float v_reset, float c_m, bool hard_reset, "
        "bool return_dense, bool return_events) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(btorch_cuda, CUDA, m) {
    m.impl("persistent_snn_forward", &persistent_snn_forward_cuda);
    m.impl("persistent_snn_forward_binned", &persistent_snn_forward_binned_cuda);
    m.impl(
        "persistent_snn_forward_spike_block",
        &persistent_snn_forward_spike_block_cuda);
}
