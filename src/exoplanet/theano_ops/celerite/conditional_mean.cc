#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(conditional_mean)(PyArrayObject* input0, PyArrayObject* input1,
                                     PyArrayObject* input2, PyArrayObject* input3,
                                     PyArrayObject* input4, PyArrayObject* input5,
                                     PyArrayObject* input6, PyArrayObject** output0) {
  using namespace exoplanet;

  // Read in U and extract the N and J dimensions
  int success = 0;
  npy_intp N, J;
  auto U_in = get_matrix_input<DTYPE_INPUT_0>(&N, &J, input0, &success);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError, "runtime value of J does not match compiled value");
    return 1;
  }

  npy_intp M, J2;
  auto U_star_in = get_matrix_input<DTYPE_INPUT_4>(&M, &J2, input4, &success);
  if (success) return 1;
  if (J2 != J) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }

  npy_intp input1_shape[] = {N, J};
  npy_intp input2_shape[] = {N - 1, J};
  npy_intp input3_shape[] = {N};
  npy_intp input5_shape[] = {M, J};
  npy_intp input6_shape[] = {M};
  auto V_in = get_input<DTYPE_INPUT_1>(2, input1_shape, input1, &success);
  auto P_in = get_input<DTYPE_INPUT_2>(2, input2_shape, input2, &success);
  auto z_in = get_input<DTYPE_INPUT_3>(1, input3_shape, input3, &success);
  auto V_star_in = get_input<DTYPE_INPUT_5>(2, input5_shape, input5, &success);
  auto inds_in = get_input<DTYPE_INPUT_6>(1, input6_shape, input6, &success);
  if (success) return 1;

  auto mu_out =
      allocate_output<DTYPE_OUTPUT_0>(1, input6_shape, TYPENUM_OUTPUT_0, output0, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> U(U_in, N,
                                                                                           J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> V(V_in, N,
                                                                                           J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> P(
      P_in, N - 1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, 1>> z(z_in, N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> U_star(
      U_star_in, M, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_5, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER>> V_star(
      V_star_in, M, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_6, Eigen::Dynamic, 1>> inds(inds_in, M);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1>> mu(mu_out, N);

  celerite::conditional_mean(U, V, P, z, U_star, V_star, inds, mu);

  return 0;
}
