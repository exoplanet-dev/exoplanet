#ifndef _KEPLER_H_
#define _KEPLER_H_

#include <Eigen/Core>
#include <cmath>

#include "exoplanet/kepler.h"

template <typename Device, typename T>
struct KeplerFunctor {
  void operator()(const Device& d, int maxiter, float tol, int size, const T* M, const T* e, T* E);
};

#if GOOGLE_CUDA
template <typename T>
struct KeplerFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int maxiter, float tol, int size, const T* M, const T* e, T* E);
};
#endif

#endif
