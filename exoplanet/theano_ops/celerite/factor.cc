#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(factor)(PyArrayObject* input0, PyArrayObject* input1,
                           PyArrayObject* input2, PyArrayObject* input3,
                           PyArrayObject** output0, PyArrayObject** output1,
                           PyArrayObject** output2, PyArrayObject** output3) {
  using namespace exoplanet;

  // Read in U and extract the N and J dimensions
  int success = 0;
  npy_intp N, J;
  auto U_in = get_matrix_input<DTYPE_INPUT_1>(&N, &J, input1, &success);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError,
                 "runtime value of J does not match compiled value");
    return 1;
  }

  // Read in the other elements
  npy_intp input0_shape[] = {N};
  npy_intp input2_shape[] = {N, J};
  npy_intp input3_shape[] = {N - 1, J};
  auto a_in = get_input<DTYPE_INPUT_0>(1, input0_shape, input0, &success);
  auto V_in = get_input<DTYPE_INPUT_2>(2, input2_shape, input2, &success);
  auto P_in = get_input<DTYPE_INPUT_3>(2, input3_shape, input3, &success);
  if (success) return 1;

  npy_intp shape0[] = {N};
  npy_intp shape1[] = {N, J};
  npy_intp shape2[] = {N, J * J};
  npy_intp shape3[] = {};
  auto d_out = allocate_output<DTYPE_OUTPUT_0>(1, shape0, TYPENUM_OUTPUT_0,
                                               output0, &success);
  auto W_out = allocate_output<DTYPE_OUTPUT_1>(2, shape1, TYPENUM_OUTPUT_1,
                                               output1, &success);
  auto S_out = allocate_output<DTYPE_OUTPUT_2>(2, shape2, TYPENUM_OUTPUT_2,
                                               output2, &success);
  auto flag_out = allocate_output<DTYPE_OUTPUT_3>(0, shape3, TYPENUM_OUTPUT_3,
                                                  output3, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> a(a_in, N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      U(U_in, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      V(V_in, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      P(P_in, N - 1, J);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1>> d(d_out, N);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, CELERITE_J,
                           CELERITE_J_ORDER>>
      W(W_out, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, CELERITE_J2,
                           CELERITE_J_ORDER>>
      S(S_out, N, J * J);

  d = a;
  W = V;
  S.setZero();
  *flag_out = celerite::factor(U, P, d, W, S);

  return 0;
}
