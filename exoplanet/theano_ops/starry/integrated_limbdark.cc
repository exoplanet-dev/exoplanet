#section support_code_struct

starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(L);

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
    PyArrayObject* input1,    // Array of impact parameters "b"
    PyArrayObject* input2,    // Array of radius ratios "r"
    PyArrayObject* input3,    // Array of line-of-sight position "los"
    PyArrayObject* input4,    //
    PyArrayObject* input5,    //
    PyArrayObject* input6,    //
    PyArrayObject** output0,  // Flux
    PyArrayObject** output1,  // dfdcl
    PyArrayObject** output2,  // dfdb
    PyArrayObject** output3,  // dfdr
    PyArrayObject** output4,  // dfdbdot
    PyArrayObject** output5,  // dfdbdotdot
    PARAMS_TYPE* params
  )
{
  typedef DTYPE_INPUT_0 Scalar;

  double tol = params->tol;
  int min_depth = params->min_depth;
  int max_depth = params->max_depth;
  bool include_contacts = params->include_contacts;

  npy_intp Nc, Nb, Nr, Nlos, Ndbdt, Nd2bdt2, Ndt;
  int success = get_size(input0, &Nc);
  success += get_size(input1, &Nb);
  success += get_size(input2, &Nr);
  success += get_size(input3, &Nlos);
  success += get_size(input4, &Ndbdt);
  success += get_size(input5, &Nd2bdt2);
  success += get_size(input6, &Ndt);
  if (success) return 1;
  if (Nb != Nr || Nb != Nlos || Nb != Ndbdt || Nb != Nd2bdt2 || Nb != Ndt) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch %d %d %d %d %d %d",
                 Nb, Nr, Nlos, Ndbdt, Nd2bdt2, Ndt);
    return 1;
  }

  npy_intp ndim = PyArray_NDIM(input1);
  npy_intp* dims = PyArray_DIMS(input1);
  std::vector<npy_intp> shape(ndim + 1);
  shape[0] = Nc;
  for (npy_intp i = 0; i < ndim; ++i) shape[i+1] = dims[i];

  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_0, output0);
  success += allocate_output(ndim+1, &(shape[0]), TYPENUM_OUTPUT_1, output1);
  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_2, output2);
  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_3, output3);
  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_4, output4);
  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_5, output5);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0* c   = (DTYPE_INPUT_0*) PyArray_DATA(input0);
  DTYPE_INPUT_1* b   = (DTYPE_INPUT_1*) PyArray_DATA(input1);
  DTYPE_INPUT_2* r   = (DTYPE_INPUT_2*) PyArray_DATA(input2);
  DTYPE_INPUT_3* los = (DTYPE_INPUT_3*) PyArray_DATA(input3);
  DTYPE_INPUT_4* bt  = (DTYPE_INPUT_4*) PyArray_DATA(input4);
  DTYPE_INPUT_5* btt = (DTYPE_INPUT_5*) PyArray_DATA(input5);
  DTYPE_INPUT_6* dt  = (DTYPE_INPUT_6*) PyArray_DATA(input6);

  DTYPE_OUTPUT_0* f      = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);
  DTYPE_OUTPUT_1* dfdcl  = (DTYPE_OUTPUT_1*)PyArray_DATA(*output1);
  DTYPE_OUTPUT_2* dfdb   = (DTYPE_OUTPUT_2*)PyArray_DATA(*output2);
  DTYPE_OUTPUT_3* dfdr   = (DTYPE_OUTPUT_3*)PyArray_DATA(*output3);
  DTYPE_OUTPUT_4* dfdbt  = (DTYPE_OUTPUT_4*)PyArray_DATA(*output4);
  DTYPE_OUTPUT_5* dfdbtt = (DTYPE_OUTPUT_5*)PyArray_DATA(*output5);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dfdcl_mat(dfdcl, Nc, Nb);
  dfdcl_mat.setZero();

  // Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  if (APPLY_SPECIFIC(L) == NULL || APPLY_SPECIFIC(L)->lmax != Nc - 1) {
    if (APPLY_SPECIFIC(L) != NULL) delete APPLY_SPECIFIC(L);
    APPLY_SPECIFIC(L) = new starry::limbdark::GreensLimbDark<Scalar>(Nc-1);
  }

  int n_grad = Nc + 4;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Grad;
  typedef Eigen::AutoDiffScalar<Grad> Diff;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec(Nc);
  for (int i = 0; i < Nc; ++i) {
    cvec(i) = c[i];
  }

  vice::integrate::simpson_adapt integrator;
  auto func = vice::functors::LimbDarkFunctor<Diff, Scalar>(APPLY_SPECIFIC(L), cvec);
  std::vector<Scalar> limits(10);


  for (npy_intp i = 0; i < Nb; ++i) {
    f[i]      = 0;
    dfdb[i]   = 0;
    dfdr[i]   = 0;
    dfdbt[i]  = 0;
    dfdbtt[i] = 0;

    if (los[i] > 0) {
      auto r_      = Diff(r[i], n_grad, 0);
      auto b_      = Diff(b[i], n_grad, 1);
      auto dbdt_   = Diff(bt[i], n_grad, 2);
      auto d2bdt2_ = Diff(btt[i], n_grad, 3);

      int nlim = func.setup(r_, b_, dbdt_, d2bdt2_, dt[i], limits, include_contacts);
      auto ym = func(limits[0]);
      for (int j = 0; j < nlim-1; ++j) {
          auto yp = func(limits[j+1]);
          auto val = integrator(func, limits[j], limits[j+1], ym, yp, tol, max_depth, min_depth) / dt[i];
          f[i] += val.value();

          auto grad = val.derivatives();
          dfdr[i]   += grad(0);
          dfdb[i]   += grad(1);
          dfdbt[i]  += grad(2);
          dfdbtt[i] += grad(3);
          dfdcl_mat.col(i) += grad.tail(grad.rows() - 4).transpose();

          ym = yp;
      }

    }
  }

  return 0;
}
