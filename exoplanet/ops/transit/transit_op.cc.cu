#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
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
                            const T*  __restrict__   const direction,
                            T*  __restrict__               delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    if (direction[i] > 0) {
      delta[i] = transit::compute_transit_depth<T>(N, radius, intensity, n_min[i], n_max[i], z[i], r[i]);
    }
  }
}

template <typename T>
void TransitDepthFunctor<T>::operator()(
    const GPUDevice& d, int N, const T* const radius, const T* const intensity,
    int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r,
    const T* const direction, T* delta)
{
  /*GPUDevice d = ctx->eigen_device<GPUDevice>();*/
  /*CudaLaunchConfig config = GetCudaLaunchConfig(size, d);*/
  int block_count = 1024;
  int thread_per_block = 20;
  TransitDepthCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(N, radius, intensity, size, n_min, n_max, z, r, direction, delta);
}

template struct TransitDepthFunctor<GPUDevice, float>;
template struct TransitDepthFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
