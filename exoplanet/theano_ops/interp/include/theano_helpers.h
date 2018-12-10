#include <Eigen/Core>

#ifndef REGULAR_GRID_NOUT
#define REGULAR_GRID_NOUT Eigen::Dynamic
#define REGULAR_GRID_NOUT_ORDER Eigen::ColMajor
#endif

int get_dimensions(PyArrayObject* input, npy_intp* N) {
  if (input == NULL || PyArray_NDIM(input) != 1 || !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  *N = PyArray_DIM(input, 0);
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
