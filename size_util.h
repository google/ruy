#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIZE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIZE_UTIL_H_

#include "check_macros.h"

namespace ruy {

inline int floor_log2(int n) {
  RUY_DCHECK_GE(n, 1);
  return 31 - __builtin_clz(n);
}

inline int ceil_log2(int n) {
  RUY_DCHECK_GE(n, 1);
  return n == 1 ? 0 : floor_log2(n - 1) + 1;
}

inline int round_down_pot(int value) { return 1 << floor_log2(value); }

inline int round_up_pot(int value) { return 1 << ceil_log2(value); }

inline int round_down_pot(int value, int modulo) {
  RUY_DCHECK_EQ(modulo & (modulo - 1), 0);
  return value & ~(modulo - 1);
}

inline int round_up_pot(int value, int modulo) {
  return round_down_pot(value + modulo - 1, modulo);
}

inline int clamp(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  } else if (x > hi) {
    return hi;
  } else {
    return x;
  }
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIZE_UTIL_H_
