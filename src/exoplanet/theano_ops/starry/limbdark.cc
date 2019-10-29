#section support_code_struct

exoplanet::starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(L);

#section init_code_struct

{ APPLY_SPECIFIC(L) = NULL; }

#section cleanup_code_struct

if (APPLY_SPECIFIC(L) != NULL) {
  delete APPLY_SPECIFIC(L);
}

#section support_code_struct

int APPLY_SPECIFIC(limbdark)(PyArrayObject* input0,    // Array of "cl"
                             PyArrayObject* input1,    // Array of impact parameters "b"
                             PyArrayObject* input2,    // Array of radius ratios "r"
                             PyArrayObject* input3,    // Array of line-of-sight position "los"
                             PyArrayObject** output0,  // Flux
                             PyArrayObject** output1,  // dfdcl
                             PyArrayObject** output2,  // dfdb
                             PyArrayObject** output3   // dfdr
) {
  using namespace exoplanet;

  int success = 0;
  int ndim_c = -1;
  npy_intp* shape_c;
  auto c = get_input<DTYPE_INPUT_0>(&ndim_c, &shape_c, input0, &success);
  if (ndim_c != 1) {
    PyErr_Format(PyExc_ValueError, "c must be 1D");
    return 1;
  }

  int ndim = -1;
  npy_intp* shape;
  auto b = get_input<DTYPE_INPUT_1>(&ndim, &shape, input1, &success);
  auto r = get_input<DTYPE_INPUT_2>(&ndim, &shape, input2, &success);
  auto los = get_input<DTYPE_INPUT_3>(&ndim, &shape, input3, &success);
  if (success) return 1;

  std::vector<npy_intp> new_shape(ndim + ndim_c);
  int Nc = 1, Nb = 1;
  for (int i = 0; i < ndim_c; ++i) {
    new_shape[i] = shape_c[i];
    Nc *= shape_c[i];
  }
  for (int i = 0; i < ndim; ++i) {
    new_shape[ndim_c + i] = shape[i];
    Nb *= shape[i];
  }

  auto f = allocate_output<DTYPE_OUTPUT_0>(ndim, shape, TYPENUM_OUTPUT_0, output0, &success);
  auto dfdcl = allocate_output<DTYPE_OUTPUT_1>(ndim + ndim_c, &(new_shape[0]), TYPENUM_OUTPUT_1,
                                               output1, &success);
  auto dfdb = allocate_output<DTYPE_OUTPUT_2>(ndim, shape, TYPENUM_OUTPUT_2, output2, &success);
  auto dfdr = allocate_output<DTYPE_OUTPUT_3>(ndim, shape, TYPENUM_OUTPUT_3, output3, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dfdcl_mat(dfdcl, Nc, Nb);
  dfdcl_mat.setZero();

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  if (APPLY_SPECIFIC(L) == NULL || APPLY_SPECIFIC(L)->lmax != Nc - 1) {
    if (APPLY_SPECIFIC(L) != NULL) delete APPLY_SPECIFIC(L);
    APPLY_SPECIFIC(L) = new starry::limbdark::GreensLimbDark<double>(Nc - 1);
  }

  for (npy_intp i = 0; i < Nb; ++i) {
    f[i] = 0;
    dfdb[i] = 0;
    dfdr[i] = 0;

    if (los[i] > 0) {
      auto b_ = std::abs(b[i]);
      auto r_ = std::abs(r[i]);
      if (b_ < 1 + r_) {
        APPLY_SPECIFIC(L)->template compute<true>(b_, r_);
        auto sT = APPLY_SPECIFIC(L)->sT;

        // The value of the light curve
        f[i] = sT.dot(cvec) - 1;

        // The gradients
        dfdcl_mat.col(i) = sT;
        dfdb[i] = sgn(b[i]) * APPLY_SPECIFIC(L)->dsTdb.dot(cvec);
        dfdr[i] = sgn(r[i]) * APPLY_SPECIFIC(L)->dsTdr.dot(cvec);
      }
    }
  }

  return 0;
}
