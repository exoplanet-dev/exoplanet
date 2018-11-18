#section support_code

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

#section support_code_apply

int APPLY_SPECIFIC(solver)(
    PyArrayObject*  input0,
    PyArrayObject*  input1,
    PyArrayObject** output0,
    PyArrayObject** output1,
    PARAMS_TYPE* params)
{
  typedef DTYPE_OUTPUT_0 T;

  long maxiter = params->maxiter;
  double tol = params->tol;

  npy_intp N, Ne;
  int success = get_size(input0, &N);
  success += get_size(input1, &Ne);
  if (success) return 1;
  if (N != Ne) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }

  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_0, output0);
  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_1, output1);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0*  M_in  = (DTYPE_INPUT_0*)PyArray_DATA(input0);
  DTYPE_INPUT_1*  e_in  = (DTYPE_INPUT_1*)PyArray_DATA(input1);
  DTYPE_OUTPUT_0* E_out = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);
  DTYPE_OUTPUT_1* f_out = (DTYPE_OUTPUT_1*)PyArray_DATA(*output1);

  for (npy_intp n = 0; n < N; ++n) {
    T M = M_in[n];
    T e = e_in[n];

    if (e >= 1) {
      PyErr_Format(PyExc_ValueError, "eccentricity must be 0 <= e < 1");
      return 1;
    }

    if (e <= tol) {
      E_out[n] = M;
      f_out[n] = wrap_into(M + M_PI, 2 * M_PI) - M_PI;
    } else {
      T E0 = M, E = M;
      T sinE, cosE, g, gp, delta, absdelta;
      for (int i = 0; i < maxiter; ++i) {
        sinE = sin(E0);
        cosE = cos(E0);
        g = E0 - e * sinE - M;
        gp = 1.0 - e * cosE;
        delta = g / (gp + tol);
        absdelta = fabs(delta);
        delta = (absdelta < T(1)) ? delta : delta / absdelta;
        E = E0 - delta;
        if (absdelta <= T(tol)) break;
        E0 = E;
      }
      E_out[n] = E;
      f_out[n] = 2.0 * atan(sqrt((1+e)/(1-e))*tan(0.5*E));
    }
  }

  return 0;
}
