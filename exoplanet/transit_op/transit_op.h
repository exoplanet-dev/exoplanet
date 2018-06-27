#ifndef _EXOPLANET_TRANSIT_OP_H_
#define _EXOPLANET_TRANSIT_OP_H_

#include <Eigen/Core>
#include "exoplanet/transit.h"

template <typename Device, typename T>
struct TransitDepthFunctor {
  void operator()(const Device& d, int N, const T* const radius, const T* const intensity,
                  int size, const int* const n_min, const int* const n_max, const T* const z, T r, T eps, T* delta);
};

template <typename Device, typename T>
struct TransitDepthRevFunctor {
  void operator()(const Device& d, int N, const T* const radius, const T* const intensity,
                  int size, const int* const n_min, const int* const n_max, const T* const z, T r, const T* const b_delta,
                  T* b_grid, T* b_z, T* b_r);
};

#if GOOGLE_CUDA
template <typename T>
struct TransitDepthFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int N, const T* const radius, const T* const intensity,
                  int size, const int* const n_min, const int* const n_max, const T* const z, T r, T eps, T* delta);
};
#endif

#endif
