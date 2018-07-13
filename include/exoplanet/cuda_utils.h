#ifndef _EXOPLANET_CUDA_UTILS_H_
#define _EXOPLANET_CUDA_UTILS_H_

#ifdef __CUDACC__
#define EXOPLANET_CUDA_CALLABLE __host__ __device__
#else
#define EXOPLANET_CUDA_CALLABLE
#endif

// From tensorflow/core/util/cuda_launch_config.h
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace exoplanet {

  inline int DivUp(int a, int b) { return (a + b - 1) / b; }

  struct CudaLaunchConfig {
    // Logical number of thread that works on the elements. If each logical
    // thread works on exactly a single element, this is the same as the working
    // element count.
    int virtual_thread_count = -1;
    // Number of threads per block.
    int thread_per_block = -1;
    // Number of blocks for Cuda kernel launch.
    int block_count = -1;
  };

  // Calculate the Cuda launch config we should use for a kernel launch.
  // This is assuming the kernel is quite simple and will largely be
  // memory-limited.
  // REQUIRES: work_element_count > 0.
  inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count,
      const Eigen::GpuDevice& d) {
    CHECK_GT(work_element_count, 0);
    CudaLaunchConfig config;
    const int virtual_thread_count = work_element_count;
    const int physical_thread_count = std::min(
        d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor(),
        virtual_thread_count);
    const int thread_per_block = std::min(1024, d.maxCudaThreadsPerBlock());
    const int block_count =
      std::min(DivUp(physical_thread_count, thread_per_block),
          d.getNumCudaMultiProcessors());

    config.virtual_thread_count = virtual_thread_count;
    config.thread_per_block = thread_per_block;
    config.block_count = block_count;
    return config;
  }

};

#endif

#endif
