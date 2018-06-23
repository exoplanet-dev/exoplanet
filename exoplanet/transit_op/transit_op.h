#ifndef _TRANSIT_TRANSIT_OP_H_
#define _TRANSIT_TRANSIT_OP_H_

#include <Eigen/Core>
#include "transit.h"

template <typename Device, typename T>
struct TransitFunctor {
  void operator()(const Device& d, int grid_size, const T* const x, const T* const grid,
                  int size, const int* const indmin, const int* const indmax, const T* const z, const T* const r, T* delta);
};

template <typename Device, typename T>
struct TransitRevFunctor {
  void operator()(const Device& d, int grid_size, const T* const x, const T* const grid,
                  int size, const int* const indmin, const int* const indmax, const T* const z, const T* const r, const T* const b_delta,
                  T* b_grid, T* b_z, T* b_r);
};

#if GOOGLE_CUDA
template <typename T>
struct TransitFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int grid_size, const T* const x, const T* const grid,
                  int size, const int* const indmin, const int* const indmax, const T* const z, const T* const r, T* delta);
};
#endif

#endif
