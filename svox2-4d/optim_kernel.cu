// Copyright 2021 Alex Yu
// Optimizer-related kernels

#include <torch/extension.h>
#include "cuda_util.cuh"

namespace {

const int RMSPROP_STEP_CUDA_THREADS = 256;
const int MIN_BLOCKS_PER_SM = 4;

namespace device {

// RMSPROP
__inline__ __device__ void rmsprop_once(
        float* __restrict__ ptr_data,
        float* __restrict__ ptr_rms,
        float* __restrict__ ptr_grad,
        const float beta, const float lr, const float epsilon, float minval) {
    float rms = *ptr_rms;
    rms = rms == 0.f ? _SQR(*ptr_grad) : lerp(_SQR(*ptr_grad), rms, beta);
    *ptr_rms = rms;
    *ptr_data = fmaxf(*ptr_data - lr * (*ptr_grad) / (sqrtf(rms) + epsilon), minval);
    *ptr_grad = 0.f;
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        float beta,
        float lr,
        float epsilon,
        float minval,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    int32_t chnl = tid % all_data.size(1);
    rmsprop_once(all_data.data() + tid,
                 all_rms.data() + tid,
                 all_grad.data() + tid,
                 beta,
                 (chnl == all_data.size(1) - 1) ? lr_last : lr,
                 epsilon,
                 minval);
}


__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_mask_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        const bool* __restrict__ mask,
        float beta,
        float lr,
        float epsilon,
        float minval,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    if (mask[tid / all_data.size(1)] == false) return;
    int32_t chnl = tid % all_data.size(1);
    rmsprop_once(all_data.data() + tid,
                 all_rms.data() + tid,
                 all_grad.data() + tid,
                 beta,
                 (chnl == all_data.size(1) - 1) ? lr_last : lr,
                 epsilon,
                 minval);
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_index_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
        float beta,
        float lr,
        float epsilon,
        float minval,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, indices.size(0) * all_data.size(1));
    int32_t i = indices[tid / all_data.size(1)];
    int32_t chnl = tid % all_data.size(1);
    size_t off = i * all_data.size(1) + chnl;
    rmsprop_once(all_data.data() + off, all_rms.data() + off,
                 all_grad.data() + off,
                 beta,
                 (chnl == all_data.size(1) - 1) ? lr_last : lr,
                 epsilon,
                 minval);
}


// SGD
__inline__ __device__ void sgd_once(
        float* __restrict__ ptr_data,
        float* __restrict__ ptr_grad,
        const float lr) {
    *ptr_data -= lr * (*ptr_grad);
    *ptr_grad = 0.f;
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void sgd_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        float lr,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    int32_t chnl = tid % all_data.size(1);
    sgd_once(all_data.data() + tid,
             all_grad.data() + tid,
             (chnl == all_data.size(1) - 1) ? lr_last : lr);
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void sgd_mask_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        const bool* __restrict__ mask,
        float lr,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
    if (mask[tid / all_data.size(1)] == false) return;
    int32_t chnl = tid % all_data.size(1);
    sgd_once(all_data.data() + tid,
             all_grad.data() + tid,
             (chnl == all_data.size(1) - 1) ? lr_last : lr);
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void sgd_index_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
        float lr,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, indices.size(0) * all_data.size(1));
    int32_t i = indices[tid / all_data.size(1)];
    int32_t chnl = tid % all_data.size(1);
    size_t off = i * all_data.size(1) + chnl;
    sgd_once(all_data.data() + off,
            all_grad.data() + off,
            (chnl == all_data.size(1) - 1) ? lr_last : lr);
}



}  // namespace device
}  // namespace

void rmsprop_step(
        torch::Tensor data,
        torch::Tensor rms,
        torch::Tensor grad,
        torch::Tensor indexer,
        float beta,
        float lr,
        float epsilon,
        float minval,
        float lr_last) {

    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(rms);
    CHECK_INPUT(grad);
    CHECK_INPUT(indexer);

    if (lr_last < 0.f) lr_last = lr;

    const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;

    if (indexer.dim() == 0) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::rmsprop_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                beta,
                lr,
                epsilon,
                minval,
                lr_last);
    } else if (indexer.size(0) == 0) {
        // Skip
    } else if (indexer.scalar_type() == at::ScalarType::Bool) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::rmsprop_mask_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.data_ptr<bool>(),
                beta,
                lr,
                epsilon,
                minval,
                lr_last);
    } else {
        const size_t Q = indexer.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::rmsprop_index_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                beta,
                lr,
                epsilon,
                minval,
                lr_last);
    }

    CUDA_CHECK_ERRORS;
}

void sgd_step(
        torch::Tensor data,
        torch::Tensor grad,
        torch::Tensor indexer,
        float lr,
        float lr_last) {

    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(grad);
    CHECK_INPUT(indexer);

    if (lr_last < 0.f) lr_last = lr;

    const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;

    if (indexer.dim() == 0) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::sgd_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                lr,
                lr_last);
    } else if (indexer.size(0) == 0) {
        // Skip
    } else if (indexer.scalar_type() == at::ScalarType::Bool) {
        const size_t Q = data.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::sgd_mask_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.data_ptr<bool>(),
                lr,
                lr_last);
    } else {
        const size_t Q = indexer.size(0) * data.size(1);
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
        device::sgd_index_step_kernel<<<blocks, cuda_n_threads>>>(
                data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
                indexer.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                lr,
                lr_last);
    }

    CUDA_CHECK_ERRORS;
}

// =============================================
// Mod 6: Anchor gradient regularization kernel
// =============================================

namespace {
namespace device {

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void anchor_grad_kernel(
        float* __restrict__ data,
        float* __restrict__ grad,
        const float* __restrict__ anchor,
        const bool* __restrict__ mask,
        int64_t N, int64_t C,
        float lambda_reg) {
    CUDA_GET_THREAD_ID(tid, N * C);
    int64_t voxel = tid / C;
    if (!mask[voxel]) return;
    grad[tid] += lambda_reg * (data[tid] - anchor[tid]);
}

// =============================================
// Mod 5: Weighted RMSprop kernel (per-voxel LR)
// =============================================

__inline__ __device__ void rmsprop_weighted_once(
        float* __restrict__ ptr_data,
        float* __restrict__ ptr_rms,
        float* __restrict__ ptr_grad,
        const float beta, const float lr, const float epsilon,
        float minval, float lr_weight) {
    float rms = *ptr_rms;
    rms = rms == 0.f ? _SQR(*ptr_grad) : lerp(_SQR(*ptr_grad), rms, beta);
    *ptr_rms = rms;
    float effective_lr = lr * lr_weight;
    *ptr_data = fmaxf(*ptr_data - effective_lr * (*ptr_grad) / (sqrtf(rms) + epsilon), minval);
    *ptr_grad = 0.f;
}

__launch_bounds__(RMSPROP_STEP_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void rmsprop_weighted_index_step_kernel(
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_data,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_rms,
        torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> all_grad,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> indices,
        const float* __restrict__ lr_weights,
        float beta,
        float lr,
        float epsilon,
        float minval,
        float lr_last) {
    CUDA_GET_THREAD_ID(tid, indices.size(0) * all_data.size(1));
    int32_t idx_pos = tid / all_data.size(1);
    int32_t i = indices[idx_pos];
    int32_t chnl = tid % all_data.size(1);
    size_t off = i * all_data.size(1) + chnl;
    float w = lr_weights[i];
    rmsprop_weighted_once(all_data.data() + off, all_rms.data() + off,
                 all_grad.data() + off,
                 beta,
                 (chnl == all_data.size(1) - 1) ? lr_last : lr,
                 epsilon,
                 minval,
                 w);
}

}  // namespace device
}  // namespace

// C++ wrapper for anchor gradient
void anchor_grad_step(
        torch::Tensor data,
        torch::Tensor grad,
        torch::Tensor anchor,
        torch::Tensor mask,
        float lambda_reg) {
    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(grad);
    CHECK_INPUT(anchor);
    CHECK_INPUT(mask);

    if (lambda_reg == 0.f) return;

    const int64_t N = data.size(0);
    const int64_t C = data.size(1);
    const size_t Q = N * C;
    const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);

    device::anchor_grad_kernel<<<blocks, cuda_n_threads>>>(
            data.data_ptr<float>(),
            grad.data_ptr<float>(),
            anchor.data_ptr<float>(),
            mask.data_ptr<bool>(),
            N, C,
            lambda_reg);
    CUDA_CHECK_ERRORS;
}

// C++ wrapper for weighted RMSprop
void rmsprop_weighted_step(
        torch::Tensor data,
        torch::Tensor rms,
        torch::Tensor grad,
        torch::Tensor indexer,
        torch::Tensor lr_weights,
        float beta,
        float lr,
        float epsilon,
        float minval,
        float lr_last) {

    DEVICE_GUARD(data);
    CHECK_INPUT(data);
    CHECK_INPUT(rms);
    CHECK_INPUT(grad);
    CHECK_INPUT(indexer);
    CHECK_INPUT(lr_weights);

    if (lr_last < 0.f) lr_last = lr;

    const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;

    if (indexer.size(0) == 0) return;

    const size_t Q = indexer.size(0) * data.size(1);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    device::rmsprop_weighted_index_step_kernel<<<blocks, cuda_n_threads>>>(
            data.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            rms.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            grad.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
            indexer.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            lr_weights.data_ptr<float>(),
            beta,
            lr,
            epsilon,
            minval,
            lr_last);
    CUDA_CHECK_ERRORS;
}
