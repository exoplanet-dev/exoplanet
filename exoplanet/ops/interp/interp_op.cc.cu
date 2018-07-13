#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "interp_op.h"
#include "exoplanet/cuda_utils.h"

using namespace exoplanet;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void InterpCUDAKernel(int size, int M, const T* const x, const T* const y, int N, const T* const t, T* v, int* inds) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    int k = i / N;
    int off_m = k * M;
    inds[i] = interp::interp1d<T>(M, x + off_m, y, t[i], v + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void InterpCUDAFunctor<T>::operator()(
    const GPUDevice& d, int size, int M, const T* const x, const T* const y, int N, const T* const t, T* v, int* inds) {
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  int block_count = config.block_count;
  int thread_per_block = config.thread_per_block;
  InterpCUDAKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, M, x, y, N, t, v, inds);
}

template struct InterpCUDAFunctor<float>;
template struct InterpCUDAFunctor<double>;

#endif  // GOOGLE_CUDA
