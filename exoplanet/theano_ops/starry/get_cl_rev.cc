#section support_code_apply

int APPLY_SPECIFIC(get_cl_rev)(
    PyArrayObject* input0,  // Array of "u" limb darkening coeffs
    PyArrayObject** output0) {
  using namespace exoplanet;
  typedef DTYPE_OUTPUT_0 T;

  int success = 0;
  npy_intp N = -1;
  auto bc_in = get_input<DTYPE_INPUT_0>(&N, input0, &success);
  if (success) return 1;

  auto bu = allocate_output<DTYPE_OUTPUT_0>(
      PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_0, output0,
      &success);
  if (success) return 1;

  Eigen::Matrix<T, Eigen::Dynamic, 1> bc(N);
  Eigen::Matrix<T, Eigen::Dynamic, 1> ba(N);

  for (npy_intp i = 0; i < N; ++i) {
    bc(i) = bc_in[i];
    bu[i] = 0;
  }

  if (N >= 3) {
    // c[0] = a(0) + 2 * c[2];
    ba(0) = bc(0);
    bc(2) += 2 * bc(0);
  } else {
    // c[0] = a(0);
    ba(0) = bc(0);
  }

  if (N >= 4) {
    // c[1] = a(1) + 3 * c[3];
    ba(1) = bc(1);
    bc(3) += 3 * bc(1);
  } else {
    // c[1] = a(1);
    ba(1) = bc(1);
  }

  for (npy_intp j = 2; j <= N - 3; ++j) {
    // c[j] = a(j) / (j + 2) + c[j + 2];
    ba(j) = bc(j) / (j + 2);
    bc(j + 2) += bc(j);
  }
  for (npy_intp j = std::max<npy_intp>(2, N - 2); j <= N - 1; ++j) {
    // c[j] = a(j) / (j + 2);
    ba(j) = bc(j) / (j + 2);
  }

  // Compute the a_n coefficients
  T bcoeff;
  int sign;
  for (npy_intp i = 1; i < N; ++i) {
    bcoeff = 1;
    sign = 1;
    for (npy_intp j = 0; j <= i; ++j) {
      // a(j) -= u[i] * bcoeff * sign;
      bu[i] -= ba(j) * bcoeff * sign;
      sign *= -1;
      bcoeff *= (T(i - j) / (j + 1));
    }
  }

  return 0;
}
