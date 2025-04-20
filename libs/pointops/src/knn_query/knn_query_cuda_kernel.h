#ifndef _KNN_QUERY_CUDA_KERNEL
#define _KNN_QUERY_CUDA_KERNEL

#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void knn_query_cuda(int m,
                    int nsample,
                    at::Tensor xyz_tensor,
                    at::Tensor new_xyz_tensor,
                    at::Tensor offset_tensor,
                    at::Tensor new_offset_tensor,
                    at::Tensor idx_tensor,
                    at::Tensor dist2_tensor);

template <typename scalar_t>
void knn_query_cuda_launcher(int m,
                             int nsample,
                             const scalar_t *xyz,
                             const scalar_t *new_xyz,
                             const int *offset,
                             const int *new_offset,
                             int *idx,
                             scalar_t *dist2);

#endif  // _KNN_QUERY_CUDA_KERNEL