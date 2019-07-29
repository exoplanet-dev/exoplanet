#section support_code_apply

int APPLY_SPECIFIC(kepler)(PyArrayObject *input0, PyArrayObject *input1,
                           PyArrayObject **output0, PyArrayObject **output1) {
  using namespace exoplanet;

  typedef DTYPE_OUTPUT_0 T;

  int success = 0;
  int ndim = -1;
  npy_intp *shape;
  auto M_in = get_input<DTYPE_INPUT_0>(&ndim, &shape, input0, &success);
  auto e_in = get_input<DTYPE_INPUT_1>(&ndim, &shape, input1, &success);
  if (success) return 1;

  auto sinf_out = allocate_output<DTYPE_OUTPUT_0>(ndim, shape, TYPENUM_OUTPUT_0,
                                                  output0, &success);
  auto cosf_out = allocate_output<DTYPE_OUTPUT_1>(ndim, shape, TYPENUM_OUTPUT_1,
                                                  output1, &success);
  if (success) return 1;

  npy_intp N = 1;
  for (int n = 0; n < ndim; ++n) N *= shape[n];
  const T tol = 1e-10;

  T M, e, E, tanf2, tanf2_2, denom, sE, cE;
  for (npy_intp n = 0; n < N; ++n) {
    M = M_in[n];
    e = e_in[n];

    if (e > 1) {
      PyErr_Format(PyExc_ValueError, "eccentricity must be 0 <= e < 1");
      return 1;
    }

    if (e <= tol) {
      // Special case for zero eccentricity
      sinf_out[n] = sin(M);
      cosf_out[n] = cos(M);
    } else {
      E = kepler::solve_kepler(M, e);
      sE = sin(E);
      cE = cos(E);

      // First, compute tan(0.5*E) = sin(E) / (1 + cos(E))
      denom = 1 + cE;
      if (denom > tol) {
        tanf2 = sqrt((1 + e) / (1 - e)) * sE / denom;  // tan(0.5*f)
        tanf2_2 = tanf2 * tanf2;

        // Then we compute sin(f) and cos(f) using:
        // sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
        // cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
        denom = 1 / (1 + tanf2_2);
        sinf_out[n] = 2 * tanf2 * denom;
        cosf_out[n] = (1 - tanf2_2) * denom;
      } else {
        // If cos(E) = -1, E = pi and tan(0.5*E) -> inf and f = E = pi
        sinf_out[n] = 0;
        cosf_out[n] = -1;
      }
    }
  }

  return 0;
}
