#ifndef _EXOPLANET_TRANSIT_OP_H_
#define _EXOPLANET_TRANSIT_OP_H_

#include <Eigen/Core>
#include "exoplanet/transit.h"

using GPUDevice = Eigen::GpuDevice;

//template <typename Device, typename T>
//struct TransitDepthFunctor {
//  void operator()(tensorflow::OpKernelContext* ctx, int N, const T* const radius, const T* const intensity,
//                  int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r,
//                  const T* const direction, T* delta);
//};

//template <typename Device, typename T>
//struct TransitDepthRevFunctor {
//  void operator()(tensorflow::OpKernelContext* ctx, int N, const T* const radius, const T* const intensity,
//                  int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r,
//                  const T* const direction,  const T* const b_delta,
//                  T* b_grid, T* b_z, T* b_r);
//};

#if GOOGLE_CUDA

template <typename T>
struct TransitDepthCUDAFunctor {
  void operator()(const GPUDevice& d,
      int N, const T* const radius, const T* const intensity,
      int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r,
      const T* const direction,  T* delta);
};

#endif

#endif
