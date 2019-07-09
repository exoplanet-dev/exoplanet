#section support_code_apply

int APPLY_SPECIFIC(kepler)(
    PyArrayObject*  input0,
    PyArrayObject*  input1,
    PyArrayObject** output0,
    PyArrayObject** output1)
{
  using namespace exoplanet;

  typedef DTYPE_OUTPUT_0 T;

  int success = 0;
  int ndim = -1;
  npy_intp* shape;
  auto M_in = get_input<DTYPE_INPUT_0>(&ndim, &shape, input0, &success);
  auto e_in = get_input<DTYPE_INPUT_0>(&ndim, &shape, input1, &success);
  if (success) return 1;

  auto E_out = allocate_output<DTYPE_OUTPUT_0>(ndim, shape, TYPENUM_OUTPUT_0, output0, &success);
  auto f_out = allocate_output<DTYPE_OUTPUT_1>(ndim, shape, TYPENUM_OUTPUT_1, output1, &success);
  if (success) return 1;

  npy_intp N = 1;
  for (int n = 0; n < ndim; ++n) N *= shape[n];

  T M, e, E, delta, func, esinE, tanE2, denom;
  for (npy_intp n = 0; n < N; ++n) {
    M = M_in[n];
    e = e_in[n];
    E_out[n] = M;
    f_out[n] = M;

    if (e > 1) {
      PyErr_Format(PyExc_ValueError, "eccentricity must be 0 <= e < 1");
      return 1;
    }

    const T tol = 1e-10;
    if (e <= tol) {

      // Special case for zero eccentricity
      E_out[n] = M;
      f_out[n] = wrap_into(M + M_PI, 2 * M_PI) - M_PI;

    } else {

      E = kepler::solve_kepler(M, e);
      E_out[n] = E;
      denom = e * (1 + cos(E));
      esinE = e * sin(E);
      if (fabs(denom) > tol) {
        tanE2 = esinE / denom;  // tan(0.5*E)
        f_out[n] = 2 * atan(sqrt((1+e)/(1-e))*tanE2);
      } else {
        f_out[n] = M_PI;
      }

    }
  }

  return 0;
}
