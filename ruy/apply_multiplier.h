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

// Provides a reference (portable, non-optimized) ApplyMultiplier function.

#ifndef RUY_RUY_APPLY_MULTIPLIER_H_
#define RUY_RUY_APPLY_MULTIPLIER_H_

#include <cstdint>
#include <type_traits>

namespace ruy {

// Applies the quantized multiplier to the `*accum` accumulator value, if
// applicable, that is, if AccumScalar==int32 and DstScalar!=int32. Otherwise,
// does nothing.
//
// This is slow, portable, 'reference' code. It should only be used in
// ReferenceMul and in Path::kStandardCpp. There isn't a point in optimizing it,
// either. Fast paths have that multiplier work done as part of the kernel,
// typically written in assembly anyway.
template <typename MulParamsType>
void ApplyMultiplier(const MulParamsType& mul_params, int row,
                     typename MulParamsType::AccumScalar* accum);

namespace detail {

// Copied from gemmlowp/fixedpoint.
// Similar to the ARM64 instruction, SQRDMULH. The name of this function
// is copied from the name of that instruction.
// Implements a fixed-point multiplication on values in Q0.31 format, i.e.
// the int32 values represent real numbers in [-1, 1), the int32 value -2^31
// represents the real number -1. The 'doubling' part of the name refers to
// the fact that this returns (up to correct rounding) a*b/2^31, not a*b/2^32
// as just 'high mul' would suggest.
std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b);

// Copied from gemmlowp/fixedpoint.
// Returns numerator/2^exponent, with correct round-to-nearest, breaking ties
// away-from-zero. That particular tie-breaking behavior is not particularly
// important in practice. When RUY_OPT(NATIVE_ROUNDING),
// optimized code paths may use whatever tie-breaking
// behavior is more friendly to the target instruction set, typically breaking
// ties upward.
std::int32_t RoundingDivideByPOT(std::int32_t numerator, int exponent);

// Copied from TF Lite code.
std::int32_t MultiplyByQuantizedMultiplier(std::int32_t x,
                                           std::int32_t quantized_multiplier,
                                           int shift);

// Helper to apply a fixed-point multiplier.  Only 'applicable' if AccumScalar
// is int32 (i.e. in all cases except floating-point) and if the destination is
// not int32 (i.e. unless the user wants to get raw accumulators).
template <typename MulParamsType,
          bool IsApplicable = std::is_same<typename MulParamsType::AccumScalar,
                                           std::int32_t>::value &&
                              !std::is_same<typename MulParamsType::DstScalar,
                                            std::int32_t>::value>
struct ApplyMultiplierImpl {};

// Specialization in non-applicable case: do nothing, just check that values
// are default.
template <typename MulParamsType>
struct ApplyMultiplierImpl<MulParamsType, false> {
  using AccumScalar = typename MulParamsType::AccumScalar;
  using DstScalar = typename MulParamsType::DstScalar;
  static void Run(const MulParamsType& mul_params, int, AccumScalar*) {
    RUY_DCHECK_EQ(mul_params.multiplier_fixedpoint(), 0);
    RUY_DCHECK_EQ(mul_params.multiplier_exponent(), 0);
  }
};

template <typename MulParamsType>
struct ApplyMultiplierImpl<MulParamsType, true> {
  using AccumScalar = typename MulParamsType::AccumScalar;
  using DstScalar = typename MulParamsType::DstScalar;
  static void Run(const MulParamsType& mul_params, int row,
                  AccumScalar* accum) {
    AccumScalar m = mul_params.multiplier_fixedpoint_perchannel()
                        ? mul_params.multiplier_fixedpoint_perchannel()[row]
                        : mul_params.multiplier_fixedpoint();
    int e = mul_params.multiplier_exponent_perchannel()
                ? mul_params.multiplier_exponent_perchannel()[row]
                : mul_params.multiplier_exponent();
    *accum = MultiplyByQuantizedMultiplier(*accum, m, e);
  }
};

}  // namespace detail

template <typename MulParamsType>
void ApplyMultiplier(const MulParamsType& mul_params, int row,
                     typename MulParamsType::AccumScalar* accum) {
  detail::ApplyMultiplierImpl<MulParamsType>::Run(mul_params, row, accum);
}

}  // namespace ruy

#endif  // RUY_RUY_APPLY_MULTIPLIER_H_
