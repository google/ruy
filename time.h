#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_

#include <cstdint>

#if !defined __aarch64__ && !defined _WIN32
#include <ctime>
#endif

namespace ruy {

inline std::uint64_t TimeNowBarrier() {
#ifdef __aarch64__
  std::uint64_t result = 0;
  asm volatile(
      "isb\n"
      "mrs %[result], CNTVCT_EL0\n"
      "isb\n"
      : [ result ] "=r"(result)::);
  return result;
#elif !defined _WIN32
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return 1e9 * t.tv_sec + t.tv_nsec;
#else
  return 0;
#endif
}

inline std::uint64_t TimeNowRelaxed() {
#ifdef __aarch64__
  std::uint64_t result = 0;
  // In the present Relaxed case, this is really equivalent to what
  // CycleClock::Now() does. See TimeNowBarrier for where it's different.
  asm volatile("mrs %[result], CNTVCT_EL0" : [ result ] "=r"(result)::);
  return result;
#else
  return TimeNowBarrier();
#endif
}

inline std::uint64_t TimeFrequency() {
#ifdef __aarch64__
  std::uint64_t result = 0;
  // Probably equivalent to what CycleClock::Frequency does.
  // Still want to keep this explicit path as long as we need an explicit
  // path anywhere (see TimeNowBarrier).
  asm volatile("mrs %[result], CNTFRQ_EL0" : [ result ] "=r"(result)::);
  return result;
#elif !defined _WIN32
  return 1e9;
#else
  return 0;
#endif
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
