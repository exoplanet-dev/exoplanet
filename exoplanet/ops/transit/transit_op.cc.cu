#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__
void TransitDepthCUDAKernel(int                            N,
                            const T*  __restrict__   const radius,
                            const T*  __restrict__   const intensity,
                            int                            size,
                            const int*  __restrict__ const n_min,
                            const int*  __restrict__ const n_max,
                            const T*  __restrict__   const z,
                            const T*  __restrict__   const r,
                            const T*  __restrict__   const direction,
                            T*  __restrict__               delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    if (direction[i] > T(0)) {
      delta[i] = transit::compute_transit_depth<T>(N, radius, intensity, n_min[i], n_max[i], z[i], r[i]);
    } else {
      delta[i] = T(0);
    }
  }
}

template <typename T>
void TransitDepthCUDAFunctor<T>::operator()(
    const GPUDevice& d, int N, const T* const radius, const T* const intensity,
    int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r,
    const T* const direction, T* delta)
{
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  int block_count = config.block_count;
  int thread_per_block = config.thread_per_block;
  TransitDepthCUDAKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(N, radius, intensity, size, n_min, n_max, z, r, direction, delta);
}

template struct TransitDepthCUDAFunctor<float>;
template struct TransitDepthCUDAFunctor<double>;

#endif  // GOOGLE_CUDA
