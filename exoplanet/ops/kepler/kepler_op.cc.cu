#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "kepler_op.h"
#include "exoplanet/cuda_utils.h"

using namespace exoplanet;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void KeplerCudaKernel(const int maxiter, const float tol, const int size, const T* const M, const T* const e, T* E) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    E[i] = kepler::solve_kepler<T>(M[i], e[i], maxiter, tol);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void KeplerCUDAFunctor<T>::operator()(
    const GPUDevice& d, int maxiter, float tol, int size, const T* const M, const T* const e, T* E) {
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  int block_count = config.block_count;
  int thread_per_block = config.thread_per_block;
  KeplerCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(maxiter, tol, size, M, e, E);
}

template struct KeplerCUDAFunctor<float>;
template struct KeplerCUDAFunctor<double>;

#endif  // GOOGLE_CUDA
