#ifndef _STARRY_THEANO_OPS_THEANO_HELPERS_H_
#define _STARRY_THEANO_OPS_THEANO_HELPERS_H_

#include <cmath>
#include <Eigen/Core>
#include "limbdark.h"

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

int get_size(PyArrayObject* input, npy_intp* size) {
  int flag = 0;
  if (input == NULL || !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "input must be C contiguous");
    return 1;
  }
  *size = PyArray_SIZE(input);
  return 0;
}

template <typename DTYPE_OUTPUT_NUM>
DTYPE_OUTPUT_NUM* allocate_output(int ndim, npy_intp* shape, int typenum, PyArrayObject** output, int* success) {
  bool flag = true;
  if (*output != NULL && PyArray_NDIM(*output) == ndim) {
    for (int n = 0; n < ndim; ++n) {
      if (PyArray_DIMS(*output)[n] != shape[n]) {
        flag = false;
        break;
      }
    }
  } else {
    flag = false;
  }

  if (!flag || !PyArray_CHKFLAGS(*output, NPY_ARRAY_C_CONTIGUOUS)) {
    Py_XDECREF(*output);
    *output = (PyArrayObject*)PyArray_EMPTY(ndim, shape, typenum, 0);
    if (!*output) {
      *success = 1;
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return NULL;
    }
  }

  *success = 0;
  return (DTYPE_OUTPUT_NUM*)PyArray_DATA(*output);
}

template <typename DTYPE_INPUT_NUM>
DTYPE_INPUT_NUM* get_input (npy_intp* Nr, PyArrayObject* input, int* flag) {
  npy_intp N;
  *flag = get_size(input, &N);
  if (*flag) return NULL;
  if (*Nr > 0 && N != *Nr) {
    *flag = 1;
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return NULL;
  }
  *flag = 0;
  *Nr = N;
  return (DTYPE_INPUT_NUM*) PyArray_DATA(input);
}

#endif  // _STARRY_THEANO_OPS_THEANO_HELPERS_H_
