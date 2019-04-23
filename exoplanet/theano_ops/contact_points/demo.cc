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

  auto solver = contact_points::ContactPointSolver<T>(
      a, e, cosw, sinw, cosi, sini);
  auto roots = solver.find_roots(L);

  std::cout << std::get<0>(roots) << "\n";
  std::cout << std::get<1>(roots) << "\n";
  std::cout << std::get<2>(roots) << "\n";

  return 0;
}
