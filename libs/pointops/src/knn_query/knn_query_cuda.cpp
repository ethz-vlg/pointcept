#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "knn_query_cuda_kernel.h"

void knn_query_cuda(
    int m, int nsample,
    at::Tensor xyz_tensor,
    at::Tensor new_xyz_tensor,
    at::Tensor offset_tensor,
    at::Tensor new_offset_tensor,
    at::Tensor idx_tensor,
    at::Tensor dist2_tensor
) {
    const int* offset = offset_tensor.data_ptr<int>();
    const int* new_offset = new_offset_tensor.data_ptr<int>();
    int* idx = idx_tensor.data_ptr<int>();

    if (xyz_tensor.scalar_type() == at::kFloat) {
        const float* xyz = xyz_tensor.data_ptr<float>();
        const float* new_xyz = new_xyz_tensor.data_ptr<float>();
        float* dist2 = dist2_tensor.data_ptr<float>();
        knn_query_cuda_launcher<float>(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
    } else if (xyz_tensor.scalar_type() == at::kHalf) {
        const at::Half* xyz = xyz_tensor.data_ptr<at::Half>();
        const at::Half* new_xyz = new_xyz_tensor.data_ptr<at::Half>();
        at::Half* dist2 = dist2_tensor.data_ptr<at::Half>();
        knn_query_cuda_launcher<at::Half>(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
    } else {
        AT_ERROR("knn_query_cuda only supports float32 and float16 (Half).");
    }
}
