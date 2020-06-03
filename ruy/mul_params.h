/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef RUY_RUY_SPEC_H_
#define RUY_RUY_SPEC_H_

#include <limits>
#include <type_traits>

#include "ruy/matrix.h"

namespace ruy {

// Our 'general' loop structure (the default) involves multi-threading and
// complicated loops aiming to optimize cache-friendliness. One may opt out of
// this and pick the 'simple' loop structure instead, which only performs well
// for small matrix sizes and only allows using one thread, in exchange for
// smaller code size.
enum class LoopStructure { kGeneral, kSimple, kAuto };

// In general we allow zero_point's to have any Scalar value. This is called
// 'asymmetric' quantization. We do take advantage of the optimization
// opportunities when zero_points happen at runtime to be 'symmetric' (e.g. the
// int8 value 0 or the uint8 value 128), but we still generate code to handle
// the general asymmetric case. By choosing kSymmetric here, one opts out of
// this and supports only the symmetric case, in exchange for smaller code size.
enum class ZeroPointSupport { kGeneral, kSymmetric };

// In general we allow all Layout's, even if we may use slow paths for some
// kinds of layouts. By choosing kRCC, one may opt out of this and
// only keep support for the simplest and most efficient combination of
// Layout's, in exchange for smaller code size. The case covered by
// kRCC is where the storage orders are exactly the following:
//    - LHS is RowMajor
//    - RHS is ColMajor
//    - Destination is ColMajor
enum class LayoutSupport { kGeneral, kRCC };

// MulParams describes all about a matrix multiplication that
// isn't encoded in the LHS, RHS and destination matrices. Some of that
// information is encoded as compile-time constants and types (for instance, the
// choice of accumulator type, AccumScalar). Some of that information is encoded
// as runtime values (for instance, the optional bias vector).
template <typename tAccumScalar, typename tDstScalar>
class MulParams /* not final, legitimate to subclass */ {
 public:
  // Accumulator type. The type of accumulators used to compute the dot-products
  // before being ultimately casted to the destination type.
  using AccumScalar = tAccumScalar;
  // The destination scalar type.
  using DstScalar = tDstScalar;

  const AccumScalar* bias() const { return bias_; }
  void set_bias(const AccumScalar* ptr) { bias_ = ptr; }
  AccumScalar multiplier_fixedpoint() const { return multiplier_fixedpoint_; }
  void set_multiplier_fixedpoint(const AccumScalar value) {
    multiplier_fixedpoint_ = value;
  }
  int multiplier_exponent() const { return multiplier_exponent_; }
  void set_multiplier_exponent(const int value) {
    multiplier_exponent_ = value;
  }
  const AccumScalar* multiplier_fixedpoint_perchannel() const {
    return multiplier_fixedpoint_perchannel_;
  }
  void set_multiplier_fixedpoint_perchannel(const AccumScalar* ptr) {
    multiplier_fixedpoint_perchannel_ = ptr;
  }
  const int* multiplier_exponent_perchannel() const {
    return multiplier_exponent_perchannel_;
  }
  void set_multiplier_exponent_perchannel(const int* ptr) {
    multiplier_exponent_perchannel_ = ptr;
  }
  DstScalar clamp_min() const { return clamp_min_; }
  void set_clamp_min(const DstScalar value) { clamp_min_ = value; }
  DstScalar clamp_max() const { return clamp_max_; }
  void set_clamp_max(const DstScalar value) { clamp_max_ = value; }

 protected:
  // The bias vector data, if not null.
  const AccumScalar* bias_ = nullptr;
  // Only for non-floating-point cases. The fixed-point part (i.e. the mantissa)
  // of the multiplier by which accumulators are multiplied before being casted
  // to the destination type.
  AccumScalar multiplier_fixedpoint_ = 0;
  // Only for non-floating-point cases. The exponent part of the aforementioned
  // multiplier.
  int multiplier_exponent_ = 0;
  // Per-channel variant of multiplier_fixedpoint. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_fixedpoint.
  const AccumScalar* multiplier_fixedpoint_perchannel_ = nullptr;
  // Per-channel variant of multiplier_exponent. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_exponent.
  //
  // Either none or both of multiplier_exponent_perchannel and
  // multiplier_fixedpoint_perchannel must be nullptr.
  const int* multiplier_exponent_perchannel_ = nullptr;
  // min clamp bound of destination values.
  DstScalar clamp_min_ = std::is_floating_point<DstScalar>::value
                             ? -std::numeric_limits<DstScalar>::infinity()
                             : std::numeric_limits<DstScalar>::lowest();
  // max clamp bound of destination values.
  DstScalar clamp_max_ = std::is_floating_point<DstScalar>::value
                             ? std::numeric_limits<DstScalar>::infinity()
                             : std::numeric_limits<DstScalar>::max();

 public:
  // See above enum LoopStructure
  static constexpr LoopStructure kLoopStructure = LoopStructure::kAuto;
  // See above enum LayoutSupport
  static constexpr LayoutSupport kLayoutSupport = LayoutSupport::kGeneral;
  // See above enum ZeroPointSupport
  static constexpr ZeroPointSupport kZeroPointSupport =
      ZeroPointSupport::kGeneral;
  // Testing-only, not meant to be used by actual users:
  // Used for testing of various kernel layouts.
  using StandardCppKernelLhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
};

template <typename tAccumScalar, typename tDstScalar>
using BasicSpec = MulParams<tAccumScalar, tDstScalar>;

}  // namespace ruy

#endif  // RUY_RUY_SPEC_H_
