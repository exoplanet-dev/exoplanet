#ifndef _EXOPLANET_THEANO_HELPERS_H_
#define _EXOPLANET_THEANO_HELPERS_H_

#include <cmath>

namespace exoplanet {

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
inline T wrap_into(T x, T period) {
  return x - period * floor(x / period);
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
DTYPE_OUTPUT_NUM* allocate_output(int ndim, npy_intp* shape, int typenum,
                                  PyArrayObject** output, int* success) {
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
DTYPE_INPUT_NUM* get_input(int* ndim, npy_intp** shape, PyArrayObject* input,
                           int* flag, bool check_order = true) {
  *flag = 1;

  if (input == NULL ||
      (check_order && !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS))) {
    PyErr_Format(PyExc_ValueError, "input must be C contiguous");
    return NULL;
  }

  // Check the dimensions
  if (*ndim >= 0 && PyArray_NDIM(input) != *ndim) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return NULL;
  }

  // Check the shape
  auto dims = PyArray_DIMS(input);
  if (*ndim >= 0) {
    for (int n = 0; n < *ndim; ++n) {
      if ((*shape)[n] >= 0 && dims[n] != (*shape)[n]) {
        PyErr_Format(PyExc_ValueError, "dimension mismatch");
        return NULL;
      }
    }
  } else {
    *ndim = PyArray_NDIM(input);
    *shape = dims;
  }

  *flag = 0;
  return (DTYPE_INPUT_NUM*)PyArray_DATA(input);
}

template <typename DTYPE_INPUT_NUM>
DTYPE_INPUT_NUM* get_input(npy_intp* size, PyArrayObject* input, int* flag,
                           bool check_order = true) {
  *flag = 1;

  if (input == NULL ||
      (check_order && !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS))) {
    PyErr_Format(PyExc_ValueError, "input must be C contiguous");
    return NULL;
  }

  // Check the dimensions
  if (*size >= 0 && PyArray_SIZE(input) != *size) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return NULL;
  }

  *size = PyArray_SIZE(input);
  *flag = 0;
  return (DTYPE_INPUT_NUM*)PyArray_DATA(input);
}

template <typename DTYPE_INPUT_NUM>
DTYPE_INPUT_NUM* get_input(int ndim, npy_intp* shape, PyArrayObject* input,
                           int* flag, bool check_order = true) {
  *flag = 1;

  if (input == NULL ||
      (check_order && !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS))) {
    PyErr_Format(PyExc_ValueError, "input must be C contiguous");
    return NULL;
  }

  // Check the dimensions
  if (PyArray_NDIM(input) != ndim) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return NULL;
  }

  // Check the shape
  auto dims = PyArray_DIMS(input);
  for (int n = 0; n < ndim; ++n) {
    if (shape[n] >= 0 && dims[n] != shape[n]) {
      PyErr_Format(PyExc_ValueError, "dimension mismatch");
      return NULL;
    }
  }

  *flag = 0;
  return (DTYPE_INPUT_NUM*)PyArray_DATA(input);
}

template <typename DTYPE_INPUT_NUM>
DTYPE_INPUT_NUM* get_matrix_input(npy_intp* N1, npy_intp* N2,
                                  PyArrayObject* input, int* flag,
                                  bool check_order = true) {
  int ndim = -1;
  npy_intp* shape;
  auto input_obj =
      get_input<DTYPE_INPUT_NUM>(&ndim, &shape, input, flag, check_order);
  if (*flag) return NULL;
  if (ndim != 2) {
    *flag = 1;
    PyErr_Format(PyExc_ValueError, "argument must be a matrix");
    return NULL;
  }
  *flag = 0;
  *N1 = shape[0];
  *N2 = shape[1];
  return input_obj;
}

}  // namespace exoplanet

#endif  // _EXOPLANET_THEANO_HELPERS_H_
