#section support_code_struct

exoplanet::starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(L);

#section init_code_struct

{
  APPLY_SPECIFIC(L) = NULL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(L) != NULL) {
  delete APPLY_SPECIFIC(L);
}

#section support_code_struct

int APPLY_SPECIFIC(integrated_limbdark)(
    PyArrayObject* input0,    // Array of "cl"
    PyArrayObject* input1,    // Array of radius ratios "r"
    PyArrayObject* input2,    // x
    PyArrayObject* input3,    // xt
    PyArrayObject* input4,    // xtt
    PyArrayObject* input5,    // y
    PyArrayObject* input6,    // yt
    PyArrayObject* input7,    // ytt
    PyArrayObject* input8,    // z
    PyArrayObject* input9,    // zt
    PyArrayObject* input10,   // dt
    PyArrayObject** output0,  // Flux
    PyArrayObject** output1,  // dfdcl
    PyArrayObject** output2,  // dfdr
    PyArrayObject** output3,  // dfdx
    PyArrayObject** output4,  // dfdxt
    PyArrayObject** output5,  // dfdxtt
    PyArrayObject** output6,  // dfdx
    PyArrayObject** output7,  // dfdxt
    PyArrayObject** output8,  // dfdxtt
    PyArrayObject** output9,  // num_eval
    PARAMS_TYPE* params
  )
{
  using namespace exoplanet;
  typedef DTYPE_INPUT_0 Scalar;

  double tol = params->tol * params->tol;
  int min_depth = params->min_depth;
  int max_depth = params->max_depth;
  bool include_contacts = params->include_contacts;

  npy_intp Nc = -1, Nr = -1;
  int success = 0;
  auto c   = get_input<DTYPE_INPUT_0>(&Nc, input0, &success);
  auto r   = get_input<DTYPE_INPUT_1>(&Nr, input1, &success);
  auto x   = get_input<DTYPE_INPUT_2>(&Nr, input2, &success);
  auto xt  = get_input<DTYPE_INPUT_3>(&Nr, input3, &success);
  auto xtt = get_input<DTYPE_INPUT_4>(&Nr, input4, &success);
  auto y   = get_input<DTYPE_INPUT_5>(&Nr, input5, &success);
  auto yt  = get_input<DTYPE_INPUT_6>(&Nr, input6, &success);
  auto ytt = get_input<DTYPE_INPUT_7>(&Nr, input7, &success);
  auto z   = get_input<DTYPE_INPUT_8>(&Nr, input8, &success);
  auto zt  = get_input<DTYPE_INPUT_9>(&Nr, input8, &success);
  auto dt  = get_input<DTYPE_INPUT_10>(&Nr, input10, &success);
  if (success) return 1;

  npy_intp ndim = PyArray_NDIM(input1);
  npy_intp* dims = PyArray_DIMS(input1);
  std::vector<npy_intp> shape(ndim + 1);
  shape[0] = Nc;
  for (npy_intp i = 0; i < ndim; ++i) shape[i+1] = dims[i];

  auto f      = allocate_output<DTYPE_OUTPUT_0>(ndim, dims, TYPENUM_OUTPUT_0, output0, &success);
  auto dfdcl  = allocate_output<DTYPE_OUTPUT_1>(ndim+1, &(shape[0]), TYPENUM_OUTPUT_1, output1, &success);
  auto dfdr   = allocate_output<DTYPE_OUTPUT_2>(ndim, dims, TYPENUM_OUTPUT_2, output2, &success);
  auto dfdx   = allocate_output<DTYPE_OUTPUT_3>(ndim, dims, TYPENUM_OUTPUT_3, output3, &success);
  auto dfdxt  = allocate_output<DTYPE_OUTPUT_4>(ndim, dims, TYPENUM_OUTPUT_4, output4, &success);
  auto dfdxtt = allocate_output<DTYPE_OUTPUT_5>(ndim, dims, TYPENUM_OUTPUT_5, output5, &success);
  auto dfdy   = allocate_output<DTYPE_OUTPUT_6>(ndim, dims, TYPENUM_OUTPUT_6, output6, &success);
  auto dfdyt  = allocate_output<DTYPE_OUTPUT_7>(ndim, dims, TYPENUM_OUTPUT_7, output7, &success);
  auto dfdytt = allocate_output<DTYPE_OUTPUT_8>(ndim, dims, TYPENUM_OUTPUT_8, output8, &success);
  npy_intp empty[] = {};
  auto num_eval = allocate_output<DTYPE_OUTPUT_9>(0, empty, TYPENUM_OUTPUT_9, output9, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dfdcl_mat(dfdcl, Nc, Nr);
  dfdcl_mat.setZero();

  // Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  if (APPLY_SPECIFIC(L) == NULL || APPLY_SPECIFIC(L)->lmax != Nc - 1) {
    if (APPLY_SPECIFIC(L) != NULL) delete APPLY_SPECIFIC(L);
    APPLY_SPECIFIC(L) = new starry::limbdark::GreensLimbDark<Scalar>(Nc-1);
  }

  int n_grad = Nc + 7;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Grad;
  typedef Eigen::AutoDiffScalar<Grad> Diff;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec(Nc);
  for (int i = 0; i < Nc; ++i) {
    cvec(i) = c[i];
  }

  vice::integrate::simpson_adapt integrator;
  auto func = vice::functors::LimbDarkFunctor<Diff, Scalar>(APPLY_SPECIFIC(L), cvec);
  std::vector<Scalar> limits(10);


  for (npy_intp i = 0; i < Nr; ++i) {
    f[i]      = 0;
    dfdr[i]   = 0;
    dfdx[i]   = 0;
    dfdxt[i]  = 0;
    dfdxtt[i] = 0;
    dfdy[i]   = 0;
    dfdyt[i]  = 0;
    dfdytt[i] = 0;

    if (z[i] > 0) {
      auto r_      = Diff(r[i], n_grad, 0);
      auto x_      = Diff(x[i], n_grad, 1);
      auto xt_     = Diff(xt[i], n_grad, 2);
      auto xtt_    = Diff(xtt[i], n_grad, 3);
      auto y_      = Diff(y[i], n_grad, 4);
      auto yt_     = Diff(yt[i], n_grad, 5);
      auto ytt_    = Diff(ytt[i], n_grad, 6);

      int nlim = func.setup(r_, x_, xt_, xtt_, y_, yt_, ytt_, z[i], zt[i], dt[i], limits, include_contacts);
      auto ym = func(limits[0]);
      for (int j = 0; j < nlim-1; ++j) {
          auto yp = func(limits[j+1]);
          auto val = integrator(func, limits[j], limits[j+1], ym, yp, tol, max_depth, min_depth) / dt[i];
          f[i] += val.value();

          auto grad = val.derivatives();
          dfdr[i]   += grad(0);
          dfdx[i]   += grad(1);
          dfdxt[i]  += grad(2);
          dfdxtt[i] += grad(3);
          dfdy[i]   += grad(4);
          dfdyt[i]  += grad(5);
          dfdytt[i] += grad(6);
          dfdcl_mat.col(i) += grad.tail(grad.rows() - 7).transpose();

          ym = yp;
      }
    }
  }

  *num_eval = func.num_eval();

  return 0;
}
