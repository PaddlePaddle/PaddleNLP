#include "timer.h"

// Make sure to compile this translation unit with the host compiler and not
// nvcc, lest you hit an internal compiler error (ICE) with GCC 10.3.0
#include <chrono>

namespace {
  std::uint64_t now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
  }
}

timer::timer() {
  t0 = now();
}

double timer::elapsed() const {
  std::uint64_t t1 = now();
  return 1.e-9*(t1 - t0);
}

double timer::reset() {
  std::uint64_t t1 = now();
  double ans = 1.e-9*(t1 - t0);
  t0 = t1;
  return ans;
}
