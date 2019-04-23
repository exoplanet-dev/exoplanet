#include <iostream>
#include <cmath>
#include "find_roots.h"

int main () {
  typedef double T;

  T L = 1.2;
  T a = 100.0;
  T e = 0.3;
  T w = 0.1;
  T incl = 0.5 * M_PI - 1e-5;

  T cosw = cos(w);
  T sinw = sin(w);
  T cosi = cos(incl);
  T sini = sin(incl);

  auto results = contact_points::find_roots(
      a, e, cosw, sinw, cosi, sini, L);

  //std::cout << flag << "\n";
  //for (int i = 0; i < 4; ++i) {
  //  std::cout << real_roots[i] << " " << imag_roots[i] << "\n";
  //}

  return 0;
}
