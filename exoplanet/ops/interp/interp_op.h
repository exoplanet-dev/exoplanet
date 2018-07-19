#ifndef _EXOPLANET_INTERP_OP_H_
#define _EXOPLANET_INTERP_OP_H_

#include <Eigen/Core>
#include "exoplanet/interp.h"

#if GOOGLE_CUDA
template <typename T>
struct InterpCUDAFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, int M, const T* const x, const T* const y, int N, const T* const t, T* v, int* inds);
};
#endif

#endif
