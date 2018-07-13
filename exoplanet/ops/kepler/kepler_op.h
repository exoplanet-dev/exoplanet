#ifndef _EXOPLANET_KEPLER_OP_H_
#define _EXOPLANET_KEPLER_OP_H_

#include <Eigen/Core>
#include "exoplanet/kepler.h"

#if GOOGLE_CUDA
template <typename T>
struct KeplerCUDAFunctor {
  void operator()(const Eigen::GpuDevice& d, int maxiter, float tol, int size, const T* const M, const T* const e, T* E);
};
#endif

#endif
