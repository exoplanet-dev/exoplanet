#section support_code_struct

exoplanet::starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(L);

#section init_code_struct

{ APPLY_SPECIFIC(L) = NULL; }

#section cleanup_code_struct

if (APPLY_SPECIFIC(L) != NULL) {
  delete APPLY_SPECIFIC(L);
}

#section support_code_struct

int APPLY_SPECIFIC(integrated_limbdark)(PyArrayObject* input0,     // cl
                                        PyArrayObject* input1,     // dt
                                        PyArrayObject* input2,     // t (defined relative to t_ref)
                                        PyArrayObject* input3,     // r
                                        PyArrayObject* input4,     // n
                                        PyArrayObject* input5,     // a * (1 - e^2)
                                        PyArrayObject* input6,     // sini
                                        PyArrayObject* input7,     // cosi
                                        PyArrayObject* input8,     // e
                                        PyArrayObject* input9,     // sinw
                                        PyArrayObject* input10,    // cosw
                                        PyArrayObject** output0,   // f
                                        PyArrayObject** output1,   // num_eval
                                        PyArrayObject** output2,   // dfdcl
                                        PyArrayObject** output3,   // dfdt
                                        PyArrayObject** output4,   // dfdr
                                        PyArrayObject** output5,   // dfdn
                                        PyArrayObject** output6,   // dfdaome2
                                        PyArrayObject** output7,   // dfdcosi
                                        PyArrayObject** output8,   // dfde
                                        PyArrayObject** output9,   // dfdsinw
                                        PyArrayObject** output10,  // dfdcosw
                                        PARAMS_TYPE* params) {
  using namespace exoplanet;
  typedef DTYPE_INPUT_0 Scalar;

  double tol = params->tol * params->tol;
  int min_depth = params->min_depth;
  int max_depth = params->max_depth;

  npy_intp Nc = -1;
  int success = 0;
  auto c = get_input<DTYPE_INPUT_0>(&Nc, input0, &success);
  if (LIMBDARK_NC != Eigen::Dynamic && Nc != LIMBDARK_NC) {
    PyErr_Format(PyExc_ValueError, "The number of cls doesn't match the compiled value");
    return 1;
  }

  npy_intp Np = -1, Nt = -1;
  // npy_intp Nt = -1;
  auto dt = get_matrix_input<DTYPE_INPUT_1>(&Np, &Nt, input1, &success);
  auto t = get_matrix_input<DTYPE_INPUT_2>(&Np, &Nt, input2, &success);

  auto r = get_input<DTYPE_INPUT_3>(&Np, input3, &success);
  auto n = get_input<DTYPE_INPUT_4>(&Np, input4, &success);
  auto aome2 = get_input<DTYPE_INPUT_5>(&Np, input5, &success);
  auto sini = get_input<DTYPE_INPUT_6>(&Np, input6, &success);
  auto cosi = get_input<DTYPE_INPUT_7>(&Np, input7, &success);
#ifndef LIMBDARK_CIRCULAR
  auto e = get_input<DTYPE_INPUT_8>(&Np, input8, &success);
  auto sinw = get_input<DTYPE_INPUT_9>(&Np, input9, &success);
  auto cosw = get_input<DTYPE_INPUT_10>(&Np, input10, &success);
#endif
  if (success) return 1;

  int ndim = 2;
  npy_intp dims[] = {Np, Nt};
  npy_intp dims_c[] = {Nc, Np, Nt};

  npy_intp empty[] = {};
  auto f = allocate_output<DTYPE_OUTPUT_0>(ndim, dims, TYPENUM_OUTPUT_0, output0, &success);
  auto num_eval =
      allocate_output<DTYPE_OUTPUT_1>(0, empty, TYPENUM_OUTPUT_1, output1, &success);
  auto dfdcl =
      allocate_output<DTYPE_OUTPUT_2>(ndim + 1, dims_c, TYPENUM_OUTPUT_2, output2, &success);
  auto dfdt = allocate_output<DTYPE_OUTPUT_3>(ndim, dims, TYPENUM_OUTPUT_3, output3, &success);
  auto dfdr = allocate_output<DTYPE_OUTPUT_4>(ndim, dims, TYPENUM_OUTPUT_4, output4, &success);
  auto dfdn = allocate_output<DTYPE_OUTPUT_5>(ndim, dims, TYPENUM_OUTPUT_5, output5, &success);
  auto dfdaome2 = allocate_output<DTYPE_OUTPUT_6>(ndim, dims, TYPENUM_OUTPUT_6, output6, &success);
  auto dfdcosi = allocate_output<DTYPE_OUTPUT_7>(ndim, dims, TYPENUM_OUTPUT_7, output7, &success);
#ifndef LIMBDARK_CIRCULAR
  auto dfde = allocate_output<DTYPE_OUTPUT_8>(ndim, dims, TYPENUM_OUTPUT_8, output8, &success);
  auto dfdsinw = allocate_output<DTYPE_OUTPUT_9>(ndim, dims, TYPENUM_OUTPUT_9, output9, &success);
  auto dfdcosw = allocate_output<DTYPE_OUTPUT_10>(ndim, dims, TYPENUM_OUTPUT_10, output10, &success);
#else
  auto dfde = allocate_output<DTYPE_OUTPUT_8>(0, empty, TYPENUM_OUTPUT_8, output8, &success);
  auto dfdsinw = allocate_output<DTYPE_OUTPUT_9>(0, empty, TYPENUM_OUTPUT_9, output9, &success);
  auto dfdcosw = allocate_output<DTYPE_OUTPUT_10>(0, empty, TYPENUM_OUTPUT_10, output10, &success);
#endif
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dfdcl_mat(dfdcl, Nc, Np*Nt);
  dfdcl_mat.setZero();

  // Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  if (APPLY_SPECIFIC(L) == NULL || APPLY_SPECIFIC(L)->lmax != Nc - 1) {
    if (APPLY_SPECIFIC(L) != NULL) delete APPLY_SPECIFIC(L);
    APPLY_SPECIFIC(L) = new starry::limbdark::GreensLimbDark<Scalar>(Nc - 1);
  }

#ifndef LIMBDARK_CIRCULAR
  const int n_grad = Nc + 8;
  const int NGRAD = LIMBDARK_NC < 0 ? Eigen::Dynamic : LIMBDARK_NC + 8;
#else
  const int n_grad = Nc + 5;
  const int NGRAD = LIMBDARK_NC < 0 ? Eigen::Dynamic : LIMBDARK_NC + 5;
#endif
  typedef Eigen::Matrix<Scalar, NGRAD, 1> Grad;
  typedef Eigen::AutoDiffScalar<Grad> Diff;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec(Nc);
  for (int i = 0; i < Nc; ++i) {
    cvec(i) = c[i];
  }

  integrate::simpson_adapt integrator;
#ifndef LIMBDARK_CIRCULAR
  integrate::LimbDarkFunctor<Diff, Scalar> func(APPLY_SPECIFIC(L), cvec);
#else
  integrate::CircLimbDarkFunctor<Diff, Scalar> func(APPLY_SPECIFIC(L), cvec);
#endif

  auto t_ = Diff(t[0], n_grad, 0);
  auto r_ = Diff(r[0], n_grad, 1);
  auto n_ = Diff(n[0], n_grad, 2);
  auto aome2_ = Diff(aome2[0], n_grad, 3);
  auto sini_ = Diff(sini[0], n_grad, 0);
  sini_.derivatives().setZero();  // This gradient is always zero
  auto cosi_ = Diff(cosi[0], n_grad, 4);

#ifndef LIMBDARK_CIRCULAR
  auto e_ = Diff(e[0], n_grad, 5);
  auto sinw_ = Diff(sinw[0], n_grad, 6);
  auto cosw_ = Diff(cosw[0], n_grad, 7);
#endif

  npy_intp ind = 0;
  for (npy_intp i = 0; i < Np; ++i) {
    r_.value() = r[i];
    n_.value() = n[i];
    aome2_.value() = aome2[i];
    sini_.value() = sini[i];
    cosi_.value() = cosi[i];

#ifndef LIMBDARK_CIRCULAR
    e_.value() = e[i];
    sinw_.value() = sinw[i];
    cosw_.value() = cosw[i];
#endif

    for (npy_intp j = 0; j < Nt; ++j) {
      t_.value() = t[ind];

      Diff xm = t_ - 0.5 * dt[ind];
      Diff xp = t_ + 0.5 * dt[ind];

#ifndef LIMBDARK_CIRCULAR
      func.set_parameters(n_, aome2_, e_, sinw_, cosw_, sini_, cosi_, r_);
      Diff val = integrator.operator()<Diff, Diff, Scalar, integrate::LimbDarkFunctor<Diff, Scalar>>(
                    func, xm, xp, tol, max_depth, min_depth) /
                dt[i];
#else
      func.set_parameters(n_, aome2_, sini_, cosi_, r_);
      Diff val = integrator.operator()<Diff, Diff, Scalar, integrate::CircLimbDarkFunctor<Diff, Scalar>>(
                    func, xm, xp, tol, max_depth, min_depth) /
                dt[i];
#endif

      f[ind] = val.value();

      // auto grad = val.derivatives();

      // dfdt[ind] = grad(0);
      // dfdr[ind] = grad(1);
      // dfdn[ind] = grad(2);
      // dfdaome2[ind] = grad(3);
      // dfdcosi[ind] = grad(4);

#ifndef LIMBDARK_CIRCULAR
      // dfde[ind] = grad(5);
      // dfdsinw[ind] = grad(6);
      // dfdcosw[ind] = grad(7);
#endif

      // dfdcl_mat.col(ind) = grad.tail(Nc).transpose();

      ind++;
    }
  }

#ifdef LIMBDARK_CIRCULAR
  *dfde = 0.0;
  *dfdsinw = 0.0;
  *dfdcosw = 0.0;
#endif

  *num_eval = func.num_eval();

  return 0;
}
