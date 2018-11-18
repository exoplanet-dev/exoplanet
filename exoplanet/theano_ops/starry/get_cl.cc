#section support_code_apply

int APPLY_SPECIFIC(get_cl)(
    PyArrayObject*  input0,  // Array of "u" limb darkening coeffs
    PyArrayObject** output0)
{
  typedef DTYPE_OUTPUT_0 T;

  npy_intp N;
  int success = get_size(input0, &N);
  if (success) return 1;

  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_0, output0);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0*  u = (DTYPE_INPUT_0*)PyArray_DATA(input0);
  DTYPE_OUTPUT_0* c = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);

  Eigen::Matrix<T, Eigen::Dynamic, 1> a(N);
  a.setZero();
  a(0) = 1;

  // Compute the a_n coefficients
  T bcoeff;
  int sign;
  for (npy_intp i = 1; i < N; ++i) {
    bcoeff = 1;
    sign = 1;
    for (npy_intp j = 0; j <= i; ++j) {
      a(j) -= u[i] * bcoeff * sign;
      sign *= -1;
      bcoeff *= (T(i - j) / (j + 1));
    }
  }

  // Now, compute the c_n coefficients
  for (npy_intp j = N - 1; j >= std::max<npy_intp>(2, N - 2); --j) {
    c[j] = a(j) / (j + 2);
  }
  for (npy_intp j = N - 3; j >= 2; --j) {
    c[j] = a(j) / (j + 2) + c[j + 2];
  }
  if (N >= 4)
    c[1] = a(1) + 3 * c[3];
  else
    c[1] = a(1);
  if (N >= 3)
    c[0] = a(0) + 2 * c[2];
  else
    c[0] = a(0);

  return 0;
}
