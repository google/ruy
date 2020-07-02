/* Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ruy/apply_multiplier.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace ruy {
namespace detail {
namespace {

// Copied from gemmlowp/fixedpoint.
// Similar to the ARM64 instruction, SQRDMULH. The name of this function
// is copied from the name of that instruction.
// Implements a fixed-point multiplication on values in Q0.31 format, i.e.
// the int32 values represent real numbers in [-1, 1), the int32 value -2^31
// represents the real number -1. The 'doubling' part of the name refers to
// the fact that this returns (up to correct rounding) a*b/2^31, not a*b/2^32
// as just 'high mul' would suggest.
std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

// Returns numerator/2^exponent, rounding to nearest, breaking ties
// upwards. That particular tie-breaking behavior is not important in practice.
// It happens to be cheap to implement in hardware and therefore, commonplace.
// In particular, it matches the behavior of ARM NEON rounding right shifts
// (RSHL with negative shift amount). By contrast, breaking ties away-from-zero
// or to-nearest-even is a little more expensive and less commonplace in SIMD
// hardware.
std::int32_t RoundingRightShift(std::int32_t numerator, int exponent) {
  // According to
  //   https://en.cppreference.com/w/cpp/language/operator_arithmetic ,
  // since C++20, "The value of a >> b is a/2^b rounded down (in other words,
  // right shift on signed a is arithmetic right shift)". While we currently
  // target C++14/17, this makes it reasonable to assume that the
  // implementation-defined behavior of a>>b with a<0 has converged to this
  // behavior on current compilers even in C++14/17 modes.
  RUY_DCHECK_GE(exponent, 0);
  RUY_DCHECK_LE(exponent, 31);
  const std::int32_t nudge = (exponent > 0) ? (1 << (exponent - 1)) : 0;
  // if numerator + nudge would overflow, do the computation as if it were 2^31.
  if (numerator > std::numeric_limits<std::int32_t>::max() - nudge) {
    RUY_DCHECK_GE(exponent, 1);  // This can't happen with exponent==0.
    return 1 << (31 - exponent);
  }
  return (numerator + nudge) >> exponent;
}

}  // namespace

// Copied from TF Lite code.
std::int32_t MultiplyByQuantizedMultiplier(std::int32_t x,
                                           std::int32_t quantized_multiplier,
                                           int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingRightShift(SaturatingRoundingDoublingHighMul(
                                x * (1 << left_shift), quantized_multiplier),
                            right_shift);
}

}  // namespace detail

}  // namespace ruy
