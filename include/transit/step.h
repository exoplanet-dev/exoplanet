#ifndef _TRANSIT_STEP_H_
#define _TRANSIT_STEP_H_

#include <cmath>

namespace transit {

  template <typename Scalar>
  class StepSize {

    public:
      StepSize (Scalar f) : f_(f) {}

      Scalar advance (const Scalar& x) const {
        return x + f_ * acos(x);
      }

    private:
      Scalar f_;
  };

};

#endif
