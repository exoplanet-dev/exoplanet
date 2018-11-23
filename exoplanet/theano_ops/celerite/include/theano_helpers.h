#include <Eigen/Core>
#include "celerite.h"

#ifndef CELERITE_J
#define CELERITE_J       Eigen::Dynamic
#define CELERITE_J2      Eigen::Dynamic
#define CELERITE_J_ORDER Eigen::RowMajor
#endif

#ifndef CELERITE_NRHS
#define CELERITE_NRHS        Eigen::Dynamic
#define CELERITE_JNRHS       Eigen::Dynamic
#define CELERITE_NRHS_ORDER  Eigen::RowMajor
#define CELERITE_JNRHS_ORDER Eigen::RowMajor
#endif


int get_dimensions(PyArrayObject* input, npy_intp* N, npy_intp* J) {
  if (input == NULL || PyArray_NDIM(input) != 2 || !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  *N = PyArray_DIMS(input)[0];
  *J = PyArray_DIMS(input)[1];
  return 0;
}

int check_input(int ndim, npy_intp* shape, PyArrayObject* input) {
  if (input == NULL || PyArray_NDIM(input) != ndim || !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  for (int n = 0; n < ndim; ++n) {
    if (PyArray_DIMS(input)[n] != shape[n]) {
      PyErr_Format(PyExc_ValueError, "dimension mismatch");
      return 1;
    }
  }
  return 0;
}

int allocate_output(int ndim, npy_intp* shape, int typenum, PyArrayObject** output) {
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
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  return 0;
}
