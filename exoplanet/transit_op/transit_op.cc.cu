#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__
void TransitDepthCudaKernel(int                            N,
                            const T*  __restrict__   const radius,
                            const T*  __restrict__   const intensity,
                            int                            size,
                            const int*  __restrict__ const n_min,
                            const int*  __restrict__ const n_max,
                            const T*  __restrict__   const z,
                            const T*  __restrict__   const r,
                            T*  __restrict__               delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    delta[i] = transit::compute_transit_depth<T>(N, radius, intensity, n_min[i], n_max[i], z[i], r[i]);
  }
}

template <typename T>
void TransitDepthFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int N, const T* const radius, const T* const intensity,
    int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r, T* delta)
{
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  TransitDepthCudaKernel<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(N, radius, intensity, size, n_min, n_max, z, r, delta);
}

template struct TransitDepthFunctor<GPUDevice, float>;
template struct TransitDepthFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
