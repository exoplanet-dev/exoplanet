#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(dot_l)(PyArrayObject* input0, PyArrayObject* input1, PyArrayObject* input2,
                          PyArrayObject* input3, PyArrayObject* input4, PyArrayObject** output0) {
  using namespace exoplanet;

  // Read in U and extract the N and J dimensions
  int success = 0;
  npy_intp N, J, N2, Nrhs;
  auto U_in = get_matrix_input<DTYPE_INPUT_0>(&N, &J, input0, &success);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError, "runtime value of J does not match compiled value");
    return 1;
  }

  auto Y_in = get_matrix_input<DTYPE_INPUT_4>(&N2, &Nrhs, input4, &success);
  if (success) return 1;
  if (N != N2) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  if (CELERITE_NRHS != Eigen::Dynamic && Nrhs != CELERITE_NRHS) {
    PyErr_Format(PyExc_ValueError, "runtime value of n_rhs does not match compiled value");
    return 1;
  }

  npy_intp input1_shape[] = {N - 1, J};
  npy_intp input2_shape[] = {N};
  npy_intp input3_shape[] = {N, J};
  auto P_in = get_input<DTYPE_INPUT_1>(2, input1_shape, input1, &success);
  auto d_in = get_input<DTYPE_INPUT_2>(1, input2_shape, input2, &success);
  auto W_in = get_input<DTYPE_INPUT_3>(2, input3_shape, input3, &success);
  if (success) return 1;

  npy_intp shape0[] = {N, Nrhs};
  auto Z_out = allocate_output<DTYPE_OUTPUT_0>(2, shape0, TYPENUM_OUTPUT_0, output0, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> U(U_in, N,
                                                                                           J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> P(
      P_in, N - 1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, 1>> d(d_in, N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> W(W_in, N,
                                                                                           J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER>> Y(
      Y_in, N, Nrhs);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER>> Z(
      Z_out, N, Nrhs);

  Z = Y;
  celerite::dotL(U, P, d, W, Z);

  return 0;
}
