#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(factor_rev)(PyArrayObject* input0, PyArrayObject* input1,
                               PyArrayObject* input2, PyArrayObject* input3,
                               PyArrayObject* input4, PyArrayObject* input5,
                               PyArrayObject* input6, PyArrayObject** output0,
                               PyArrayObject** output1, PyArrayObject** output2,
                               PyArrayObject** output3) {
  using namespace exoplanet;

  // Read in U and extract the N and J dimensions
  int success = 0;
  npy_intp N, J;
  auto U_in = get_matrix_input<DTYPE_INPUT_0>(&N, &J, input0, &success);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError,
                 "runtime value of J does not match compiled value");
    return 1;
  }

  npy_intp input1_shape[] = {N - 1, J};
  npy_intp input2_shape[] = {N};
  npy_intp input3_shape[] = {N, J};
  npy_intp input4_shape[] = {N, J * J};
  npy_intp input5_shape[] = {N};
  npy_intp input6_shape[] = {N, J};
  auto P_in = get_input<DTYPE_INPUT_1>(2, input1_shape, input1, &success);
  auto d_in = get_input<DTYPE_INPUT_2>(1, input2_shape, input2, &success);
  auto W_in = get_input<DTYPE_INPUT_3>(2, input3_shape, input3, &success);
  auto S_in = get_input<DTYPE_INPUT_4>(2, input4_shape, input4, &success);
  auto bd_in = get_input<DTYPE_INPUT_5>(1, input5_shape, input5, &success);
  auto bW_in = get_input<DTYPE_INPUT_6>(2, input6_shape, input6, &success);
  if (success) return 1;

  npy_intp shape0[] = {N};
  npy_intp shape1[] = {N, J};
  npy_intp shape2[] = {N, J};
  npy_intp shape3[] = {N - 1, J};
  auto ba_out = allocate_output<DTYPE_OUTPUT_0>(1, shape0, TYPENUM_OUTPUT_0,
                                                output0, &success);
  auto bU_out = allocate_output<DTYPE_OUTPUT_1>(2, shape1, TYPENUM_OUTPUT_1,
                                                output1, &success);
  auto bV_out = allocate_output<DTYPE_OUTPUT_2>(2, shape2, TYPENUM_OUTPUT_2,
                                                output2, &success);
  auto bP_out = allocate_output<DTYPE_OUTPUT_3>(2, shape3, TYPENUM_OUTPUT_3,
                                                output3, &success);
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
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, CELERITE_J2,
                           CELERITE_J_ORDER>>
      S(S_in, N, J * J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_5, Eigen::Dynamic, 1>> bd(bd_in, N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_6, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bW(bW_in, N, J);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1>> ba(ba_out, N);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bU(bU_out, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bV(bV_out, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_3, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      bP(bP_out, N - 1, J);

  ba = bd;
  bU.setZero();
  bV = bW;
  bP.setZero();
  celerite::factor_grad(U, P, d, W, S, bU, bP, ba, bV);

  return 0;
}
