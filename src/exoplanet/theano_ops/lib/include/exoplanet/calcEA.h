#ifndef _EXOPLANET_KEPLER_H_
#define _EXOPLANET_KEPLER_H_

/*
 * C code to compute the eccentric anomaly, its sine and cosine, from
 * an input mean anomaly and eccentricity.  Timothy D. Brandt wrote
 * the code; the algorithm is based on Raposo-Pulido & Pelaez, 2017,
 * MNRAS, 467, 1702.  Computational cost is equivalent to around 3
 * trig calls (sine, cosine) in tests as of August 2020.  This can be
 * further reduced if using many mean anomalies at fixed eccentricity;
 * the code would need some modest refactoring in that case.  Accuracy
 * should be within a factor of a few of machine epsilon in E-ecc*sinE
 * and in cosE up to at least ecc=0.999999.
 */

#include <cmath>
#include <cstdlib>

namespace exoplanet {
namespace calcEA {

const double one_sixth = 1. / 6;

const double if3 = 1. / 6;
const double if5 = 1. / (6. * 20);
const double if7 = 1. / (6. * 20 * 42);
const double if9 = 1. / (6. * 20 * 42 * 72);
const double if11 = 1. / (6. * 20 * 42 * 72 * 110);
const double if13 = 1. / (6. * 20 * 42 * 72 * 110 * 156);
const double if15 = 1. / (6. * 20 * 42 * 72 * 110 * 156 * 210);

const double pi = 3.14159265358979323846264338327950288;
const double pi_d_12 = 3.14159265358979323846264338327950288 / 12;
const double pi_d_6 = 3.14159265358979323846264338327950288 / 6;
const double pi_d_4 = 3.14159265358979323846264338327950288 / 4;
const double pi_d_3 = 3.14159265358979323846264338327950288 / 3;
const double fivepi_d_12 = 3.14159265358979323846264338327950288 * 5. / 12;
const double pi_d_2 = 3.14159265358979323846264338327950288 / 2;
const double sevenpi_d_12 = 3.14159265358979323846264338327950288 * 7. / 12;
const double twopi_d_3 = 3.14159265358979323846264338327950288 * 2. / 3;
const double threepi_d_4 = 3.14159265358979323846264338327950288 * 3. / 4;
const double fivepi_d_6 = 3.14159265358979323846264338327950288 * 5. / 6;
const double elevenpi_d_12 = 3.14159265358979323846264338327950288 * 11. / 12;
const double twopi = 3.14159265358979323846264338327950288 * 2;

/*
 * Evaluate sine with a series expansion.  We can guarantee that the
 * argument will be <=pi/4, and this reaches double precision (within
 * a few machine epsilon) at a signficantly lower cost than the
 * function call to sine that obeys the IEEE standard.
 */

inline double shortsin(double x) {
  double x2 = x * x;
  return x *
         (1 - x2 * (if3 -
                    x2 * (if5 - x2 * (if7 - x2 * (if9 - x2 * (if11 - x2 * (if13 - x2 * if15)))))));
}

/*
 * Modulo 2pi: works best when you use an increment so that the
 * argument isn't too much larger than 2pi.
 */
double MAmod(double M) {
  if (M < twopi && M >= 0) return M;

  if (M > twopi) {
    M -= twopi;
    if (M > twopi)
      return fmod(M, twopi);
    else
      return M;
  } else {
    M += twopi;
    if (M < 0)
      return fmod(M, twopi) + twopi;
    else
      return M;
  }
}

/*
 * Use the second-order series expanion in Raposo-Pulido & Pelaez
 * (2017) in the singular corner (eccentricity close to 1, mean
 * anomaly close to zero).
 */

double EAstart(double M, double ecc) {
  double ome = 1. - ecc;
  double sqrt_ome = sqrt(ome);

  double chi = M / (sqrt_ome * ome);
  double Lam = sqrt(8 + 9 * chi * chi);
  double S = cbrt(Lam + 3 * chi);
  double sigma = 6 * chi / (2 + S * S + 4. / (S * S));
  double s2 = sigma * sigma;

  double denom = s2 + 2;
  double E = sigma * (1 + s2 * ome *
                              ((s2 + 20) / (60. * denom) +
                               s2 * ome * (s2 * s2 * s2 + 25 * s2 * s2 + 340 * s2 + 840) /
                                   (1400 * denom * denom * denom)));

  return E * sqrt_ome;
}

/*
 * Calculate the eccentric anomaly, its sine and cosine, using a
 * variant of the algorithm suggested in Raposo-Pulido & Pelaez (2017)
 * and used in Brandt et al. (2020).  Use the series expansion above
 * to generate an initial guess in the singular corner and use a
 * fifth-order polynomial to get the initial guess otherwise.  Use
 * series and square root calls to evaluate sine and cosine, and
 * update their values using series.  Accurate to better than 1e-15 in
 * E-ecc*sin(E)-M at all mean anomalies and at eccentricies up to
 * 0.999999.
 */

void calcEA(double M, double ecc, double *E, double *sinE, double *cosE) {
  double g2s_e = 0.2588190451025207623489 * ecc;
  double g3s_e = 0.5 * ecc;
  double g4s_e = 0.7071067811865475244008 * ecc;
  double g5s_e = 0.8660254037844386467637 * ecc;
  double g6s_e = 0.9659258262890682867497 * ecc;

  double bounds[13];
  double EA_tab[9];

  int k;
  double MA, EA, sE, cE, x, y;
  double B0, B1, B2, dx, idx;
  int MAsign = 1;
  double one_over_ecc = 1e17;
  if (ecc > 1e-17) one_over_ecc = 1. / ecc;

  MA = MAmod(M);
  if (MA > pi) {
    MAsign = -1;
    MA = twopi - MA;
  }
  /* Series expansion */
  if (2 * MA + 1 - ecc < 0.2) {
    EA = EAstart(MA, ecc);
  } else {
    /* Polynomial boundaries given in Raposo-Pulido & Pelaez */
    bounds[0] = 0;
    bounds[1] = pi_d_12 - g2s_e;
    bounds[2] = pi_d_6 - g3s_e;
    bounds[3] = pi_d_4 - g4s_e;
    bounds[4] = pi_d_3 - g5s_e;
    bounds[5] = fivepi_d_12 - g6s_e;
    bounds[6] = pi_d_2 - ecc;
    bounds[7] = sevenpi_d_12 - g6s_e;
    bounds[8] = twopi_d_3 - g5s_e;
    bounds[9] = threepi_d_4 - g4s_e;
    bounds[10] = fivepi_d_6 - g3s_e;
    bounds[11] = elevenpi_d_12 - g2s_e;
    bounds[12] = pi;

    /* Which interval? */
    for (k = 11; k > 0; k--) {
      if (MA > bounds[k]) break;
    }
    // if (k < 0) k = 0;

    /* Values at the two endpoints. */

    EA_tab[0] = k * pi_d_12;
    EA_tab[6] = (k + 1) * pi_d_12;

    /* First two derivatives at the endpoints. Left endpoint first. */

    int sign = (k >= 6) ? 1 : -1;

    x = 1 / (1 - ((6 - k) * pi_d_12 + sign * bounds[abs(6 - k)]));
    y = -0.5 * (k * pi_d_12 - bounds[k]);
    EA_tab[1] = x;
    EA_tab[2] = y * x * x * x;

    x = 1 / (1 - ((5 - k) * pi_d_12 + sign * bounds[abs(5 - k)]));
    y = -0.5 * ((k + 1) * pi_d_12 - bounds[k + 1]);
    EA_tab[7] = x;
    EA_tab[8] = y * x * x * x;

    /* Solve a matrix equation to get the rest of the coefficients. */

    idx = 1 / (bounds[k + 1] - bounds[k]);

    B0 = idx * (-EA_tab[2] - idx * (EA_tab[1] - idx * pi_d_12));
    B1 = idx * (-2 * EA_tab[2] - idx * (EA_tab[1] - EA_tab[7]));
    B2 = idx * (EA_tab[8] - EA_tab[2]);

    EA_tab[3] = B2 - 4 * B1 + 10 * B0;
    EA_tab[4] = (-2 * B2 + 7 * B1 - 15 * B0) * idx;
    EA_tab[5] = (B2 - 3 * B1 + 6 * B0) * idx * idx;

    /* Now use the coefficients of this polynomial to get the initial guess. */

    dx = MA - bounds[k];
    EA =
        EA_tab[0] +
        dx * (EA_tab[1] + dx * (EA_tab[2] + dx * (EA_tab[3] + dx * (EA_tab[4] + dx * EA_tab[5]))));
  }

  /* Sine and cosine initial guesses using series */

  if (EA < pi_d_4) {
    sE = shortsin(EA);
    cE = sqrt(1 - sE * sE);
  } else if (EA > threepi_d_4) {
    sE = shortsin(pi - EA);
    cE = -sqrt(1 - sE * sE);
  } else {
    cE = shortsin(pi_d_2 - EA);
    sE = sqrt(1 - cE * cE);
  }

  double num, denom, dEA;

  /* Halley's method to update E */

  num = (MA - EA) * one_over_ecc + sE;
  denom = one_over_ecc - cE;
  dEA = num * denom / (denom * denom + 0.5 * sE * num);

  /* Use series to update sin and cos */

  if (ecc < 0.78 || MA > 0.4) {
    *E = MAsign * (EA + dEA);
    *sinE = MAsign * (sE * (1 - 0.5 * dEA * dEA) + dEA * cE);
    *cosE = cE * (1 - 0.5 * dEA * dEA) - dEA * sE;

  } else {
    /*
     * Use Householder's third order method to guarantee performance
     * in the singular corner.
     */

    dEA = num / (denom + dEA * (0.5 * sE + one_sixth * cE * dEA));
    *E = MAsign * (EA + dEA);
    *sinE = MAsign * (sE * (1 - 0.5 * dEA * dEA) + dEA * cE * (1 - dEA * dEA * one_sixth));
    *cosE = cE * (1 - 0.5 * dEA * dEA) - dEA * sE * (1 - dEA * dEA * one_sixth);
  }

  return;
}

double solve_kepler(double M, double ecc, double *cosf, double *sinf) {
  double E;
  calcEA(M, ecc, &E, sinf, cosf);

  double denom = 1 + *cosf;
  if (denom > 1.0e-10) {
    double tanf2 = sqrt((1 + ecc) / (1 - ecc)) * (*sinf) / denom;  // tan(0.5*f)
    double tanf2_2 = tanf2 * tanf2;

    // Then we compute sin(f) and cos(f) using:
    // sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    // cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tanf2_2);
    *sinf = 2 * tanf2 * denom;
    *cosf = (1 - tanf2_2) * denom;
  } else {
    // If cos(E) = -1, E = pi and tan(0.5*E) -> inf and f = E = pi
    *sinf = 0;
    *cosf = -1;
  }
  return E;
}

}  // namespace calcEA
}  // namespace exoplanet

#endif
