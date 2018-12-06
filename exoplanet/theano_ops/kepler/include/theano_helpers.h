#include <math.h>

template <typename T>
inline int sign (T x) {
  return (x > 0) - (x < 0);
}

inline double wrap_into (double x, double period) {
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
