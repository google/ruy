#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_

#include <chrono>  // NOLINT(build/c++11)

namespace ruy {

using Clock = std::chrono::steady_clock;

using TimePoint = Clock::time_point;
using Duration = Clock::duration;

inline double ToSeconds(Duration d) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
}

inline Duration DurationFromSeconds(double s) {
  return std::chrono::duration_cast<Duration>(std::chrono::duration<double>(s));
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
