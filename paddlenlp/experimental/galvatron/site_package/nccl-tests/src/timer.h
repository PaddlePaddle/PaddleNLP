#ifndef _408319ecdd5b47b28bf8f511c4fdf816
#define _408319ecdd5b47b28bf8f511c4fdf816

#include <cstdint>

// Can't include <chrono> because of bug with gcc 10.3.0
class timer {
  std::uint64_t t0;
public:
  timer();
  double elapsed() const;
  double reset();
};

#endif
