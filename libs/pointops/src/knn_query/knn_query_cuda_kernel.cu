#include "../cuda_utils.h"
#include "knn_query_cuda_kernel.h"
#include <cuda_fp16.h>

namespace knn_query_utils {

template <typename DType>
__device__ void swap(DType* x, DType* y) {
    DType tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ void reheap(float* dist, int* idx, int k) {
    int root = 0;
    int child = root * 2 + 1;
    while (child < k) {
        if (child + 1 < k && dist[child + 1] > dist[child])
            child++;
        if (dist[root] > dist[child])
            return;
        swap<float>(&dist[root], &dist[child]);
        swap<int>(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}

__device__ void heap_sort(float* dist, int* idx, int k) {
    for (int i = k - 1; i > 0; i--) {
        swap<float>(&dist[0], &dist[i]);
        swap<int>(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}

__device__ int get_bt_idx(int idx, const int* offset) {
    int i = 0;
    while (true) {
        if (idx < offset[i])
            break;
        i++;
    }
    return i;
}

template <typename scalar_t>
__device__ float load_as_float(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ float load_as_float<at::Half>(const at::Half* ptr) {
    return __half2float(*reinterpret_cast<const __half*>(ptr));
}

}  // namespace knn_query_utils

template <typename scalar_t>
__global__ void knn_query_cuda_kernel_template(
    int m, int nsample,
    const scalar_t* __restrict__ xyz,
    const scalar_t* __restrict__ new_xyz,
    const int* __restrict__ offset,
    const int* __restrict__ new_offset,
    int* __restrict__ idx,
    scalar_t* __restrict__ dist2_out) {

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    const scalar_t* this_query = new_xyz + pt_idx * 3;
    int* this_idx = idx + pt_idx * nsample;
    scalar_t* this_dist2 = dist2_out + pt_idx * nsample;

    int bt_idx = knn_query_utils::get_bt_idx(pt_idx, new_offset);
    int start = (bt_idx == 0) ? 0 : offset[bt_idx - 1];
    int end = offset[bt_idx];

    float new_x = knn_query_utils::load_as_float(this_query + 0);
    float new_y = knn_query_utils::load_as_float(this_query + 1);
    float new_z = knn_query_utils::load_as_float(this_query + 2);

    float best_dist[128];
    int best_idx[128];
    for (int i = 0; i < nsample; i++) {
        best_dist[i] = 1e10f;
        best_idx[i] = -1;
    }

    for (int i = start; i < end; i++) {
        float x = knn_query_utils::load_as_float(xyz + i * 3 + 0);
        float y = knn_query_utils::load_as_float(xyz + i * 3 + 1);
        float z = knn_query_utils::load_as_float(xyz + i * 3 + 2);
        float d2 = (new_x - x) * (new_x - x)
                 + (new_y - y) * (new_y - y)
                 + (new_z - z) * (new_z - z);
        if (d2 < best_dist[0]) {
            best_dist[0] = d2;
            best_idx[0] = i;
            knn_query_utils::reheap(best_dist, best_idx, nsample);
        }
    }

    knn_query_utils::heap_sort(best_dist, best_idx, nsample);

    for (int i = 0; i < nsample; i++) {
        this_idx[i] = best_idx[i];
        this_dist2[i] = static_cast<scalar_t>(best_dist[i]);
    }
}

template <typename scalar_t>
void knn_query_cuda_launcher(
    int m, int nsample,
    const scalar_t* xyz,
    const scalar_t* new_xyz,
    const int* offset,
    const int* new_offset,
    int* idx,
    scalar_t* dist2) {

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    knn_query_cuda_kernel_template<scalar_t><<<blocks, threads, 0>>>(
        m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2
    );
}

// Explicit instantiation
template void knn_query_cuda_launcher<float>(int, int, const float*, const float*, const int*, const int*, int*, float*);
template void knn_query_cuda_launcher<at::Half>(int, int, const at::Half*, const at::Half*, const int*, const int*, int*, at::Half*);