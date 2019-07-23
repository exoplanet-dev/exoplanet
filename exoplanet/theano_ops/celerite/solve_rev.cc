#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(solve_rev)(PyArrayObject* input0, PyArrayObject* input1,
                              PyArrayObject* input2, PyArrayObject* input3,
                              PyArrayObject* input4, PyArrayObject* input5,
                              PyArrayObject* input6, PyArrayObject* input7,
                              PyArrayObject** output0, PyArrayObject** output1,
                              PyArrayObject** output2, PyArrayObject** output3,
                              PyArrayObject** output4) {
  using namespace exoplanet;

  // Read in U and extract the N and J dimensions
  int success = 0;
  npy_intp N, J, N2, Nrhs;
  auto U_in = get_matrix_input<DTYPE_INPUT_0>(&N, &J, input0, &success);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError,
                 "runtime value of J does not match compiled value");
    return 1;
  }

  auto Z_in = get_matrix_input<DTYPE_INPUT_4>(&N2, &Nrhs, input4, &success);
  if (success) return 1;
  if (N != N2) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  if (CELERITE_NRHS != Eigen::Dynamic && Nrhs != CELERITE_NRHS) {
    PyErr_Format(PyExc_ValueError,
                 "runtime value of n_rhs does not match compiled value");
    return 1;
  }

  npy_intp input1_shape[] = {N - 1, J};
  npy_intp input2_shape[] = {N};
  npy_intp input3_shape[] = {N, J};
  npy_intp input5_shape[] = {N, J * Nrhs};
  npy_intp input6_shape[] = {N, J * Nrhs};
  npy_intp input7_shape[] = {N, Nrhs};
  auto P_in = get_input<DTYPE_INPUT_1>(2, input1_shape, input1, &success);
  auto d_in = get_input<DTYPE_INPUT_2>(1, input2_shape, input2, &success);
  auto W_in = get_input<DTYPE_INPUT_3>(2, input3_shape, input3, &success);
  auto F_in = get_input<DTYPE_INPUT_5>(2, input5_shape, input5, &success);
  auto G_in = get_input<DTYPE_INPUT_6>(2, input6_shape, input6, &success);
  auto bZ_in = get_input<DTYPE_INPUT_7>(2, input7_shape, input7, &success);
  if (success) return 1;

  npy_intp shape0[] = {N, J};
  npy_intp shape1[] = {N - 1, J};
  npy_intp shape2[] = {N};
  npy_intp shape3[] = {N, J};
  npy_intp shape4[] = {N, Nrhs};
  auto bU_out = allocate_output<DTYPE_OUTPUT_0>(2, shape0, TYPENUM_OUTPUT_0,
                                                output0, &success);
  auto bP_out = allocate_output<DTYPE_OUTPUT_1>(2, shape1, TYPENUM_OUTPUT_1,
                                                output1, &success);
  auto bd_out = allocate_output<DTYPE_OUTPUT_2>(1, shape2, TYPENUM_OUTPUT_2,
                                                output2, &success);
  auto bW_out = allocate_output<DTYPE_OUTPUT_3>(2, shape3, TYPENUM_OUTPUT_3,
                                                output3, &success);
  auto bY_out = allocate_output<DTYPE_OUTPUT_4>(2, shape4, TYPENUM_OUTPUT_4,
                                                output4, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      U(U_in, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      P(P_in, N - 1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, 1>> d(d_in, N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      W(W_in, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, CELERITE_NRHS,
                           CELERITE_NRHS_ORDER>>
      Z(Z_in, N, Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_5, Eigen::Dynamic, CELERITE_JNRHS,
                           CELERITE_JNRHS_ORDER>>
      F(F_in, N, J * Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_6, Eigen::Dynamic, CELERITE_JNRHS,
                           CELERITE_JNRHS_ORDER>>
      G(G_in, N, J * Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_7, Eigen::Dynamic, CELERITE_NRHS,
                           CELERITE_NRHS_ORDER>>
      bZ(bZ_in, N, Nrhs);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bU(bU_out, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bP(bP_out, N - 1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, 1>> bd(bd_out, N);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_3, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bW(bW_out, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_4, Eigen::Dynamic, CELERITE_NRHS,
                           CELERITE_NRHS_ORDER>>
      bY(bY_out, N, Nrhs);

  bU.setZero();
  bP.setZero();
  bd.setZero();
  bW.setZero();
  bY.setZero();
  celerite::solve_grad(U, P, d, W, Z, F, G, bZ, bU, bP, bd, bW, bY);

  return 0;
}
