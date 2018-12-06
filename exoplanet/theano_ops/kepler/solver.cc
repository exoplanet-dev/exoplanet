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

  T M0, M, e, E, delta, func, esinE, tanE2, denom;
  for (npy_intp n = 0; n < N; ++n) {
    M = M_in[n];
    e = e_in[n];
    E_out[n] = M;
    f_out[n] = M;

    if (e > 1) {
      PyErr_Format(PyExc_ValueError, "eccentricity must be 0 <= e < 1");
      return 1;
    }

    if (e <= tol) {

      // Special case for zero eccentricity
      E_out[n] = M;
      f_out[n] = wrap_into(M + M_PI, 2 * M_PI) - M_PI;

    } else {
      // Shift M for numerical stability
      M0 = 2*M_PI*floor(M / (2*M_PI));
      M -= M0;

      E = M + e*sin(M);
      esinE = e * sin(E);

      if (e < 0.99) {
        for (long n = 0; n < maxiter; ++n) {
          delta = func / (1 - e * cos(E));
          E -= delta;
          esinE = e * sin(E);
          func = E - esinE - M;
          if (fabs(func) <= tol) break;
        }
      } else {
        // Use a more robust method for large eccentricity
        for (long n = 0; n < maxiter; ++n) {
          delta = func / (1 - e * cos(E));

          // If the change is large, that normally means that the gradient is
          // close to zero. We can jump out of this region with a move of Pi
          // in the right direction.
          if (fabs(delta) < M_PI) {
            E -= delta;
          } else {
            E -= M_PI * sign(delta);
          }

          esinE = e * sin(E);
          func = E - esinE - M;
          if (fabs(func) <= tol) break;
        }
      }

      // Save the result and compute the true anomaly
      E_out[n] = E + M0;
      denom = e * (1 + cos(E));
      if (fabs(denom) > tol) {
        tanE2 = esinE / denom;  // tan(0.5*E)
        f_out[n] = 2 * atan(sqrt((1+e)/(1-e))*tanE2);
      } else {
        if (fabs(esinE) > tol)
          f_out[n] = M_PI * sign(esinE);
        else
          f_out[n] = 0;
      }

    }
  }

  return 0;
}
