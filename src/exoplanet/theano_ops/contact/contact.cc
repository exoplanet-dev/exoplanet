#section support_code_apply

int APPLY_SPECIFIC(contact)(PyArrayObject* input0, PyArrayObject* input1,
                            PyArrayObject* input2, PyArrayObject* input3,
                            PyArrayObject* input4, PyArrayObject* input5,
                            PyArrayObject* input6, PyArrayObject** output0,
                            PyArrayObject** output1, PyArrayObject** output2,
                            PARAMS_TYPE* params) {
  using namespace exoplanet;
  typedef DTYPE_OUTPUT_0 T;

  double tol = params->tol;

  int success = 0;
  int ndim = -1;
  npy_intp* shape;
  auto a = get_input<DTYPE_INPUT_0>(&ndim, &shape, input0, &success);

  npy_intp N = 1;
  for (int n = 0; n < ndim; ++n) N *= shape[n];

  auto e = get_input<DTYPE_INPUT_1>(&N, input1, &success);
  auto cosw = get_input<DTYPE_INPUT_2>(&N, input2, &success);
  auto sinw = get_input<DTYPE_INPUT_3>(&N, input3, &success);
  auto cosi = get_input<DTYPE_INPUT_4>(&N, input4, &success);
  auto sini = get_input<DTYPE_INPUT_5>(&N, input5, &success);
  auto L = get_input<DTYPE_INPUT_6>(&N, input6, &success);
  if (success) return 1;

  auto M_left = allocate_output<DTYPE_OUTPUT_0>(ndim, shape, TYPENUM_OUTPUT_0,
                                                output0, &success);
  auto M_right = allocate_output<DTYPE_OUTPUT_1>(ndim, shape, TYPENUM_OUTPUT_1,
                                                 output1, &success);
  auto flag = allocate_output<DTYPE_OUTPUT_2>(ndim, shape, TYPENUM_OUTPUT_2,
                                              output2, &success);
  if (success) return 1;

  for (npy_intp n = 0; n < N; ++n) {
    auto const solver = contact_points::ContactPointSolver<T>(
        a[n], e[n], cosw[n], sinw[n], cosi[n], sini[n]);
    auto const roots = solver.find_roots(L[n], tol);
    flag[n] = std::get<0>(roots);
    M_left[n] = std::get<1>(roots);
    M_right[n] = std::get<2>(roots);
  }

  return 0;
}
