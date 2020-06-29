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

#ifndef RUY_RUY_MUL_PARAMS_H_
#define RUY_RUY_MUL_PARAMS_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ruy/check_macros.h"

namespace ruy {

// Enumeration to designate which dimension is the 'channels', for MulParams
// features that are 'per-channel', namely the bias-vector and the quantized
// multiplier.
enum class ChannelDimension : std::int8_t {
  // kRow means that 'per-channel' means 'per row of the destination matrix'
  kRow,
  // kCol means that 'per-channel' means 'per column of the destination matrix'
  kCol
};

namespace detail {
template <typename tAccumScalar, typename tDstScalar>
struct MulParamsStorage;
}

// MulParams describes all about a matrix multiplication that
// isn't encoded in the LHS, RHS and destination matrices. Some of that
// information is encoded as compile-time constants and types (for instance, the
// choice of accumulator type, AccumScalar). Some of that information is encoded
// as runtime values (for instance, the optional bias vector).
//
// Template parameters:
// AccumScalar: Accumulator type. The type of accumulators used to compute the
// dot-products before being ultimately casted to the destination type.
// DstScalar: The destination scalar type.
template <typename tAccumScalar, typename tDstScalar>
class MulParams final {
 public:
  using AccumScalar = tAccumScalar;
  using DstScalar = tDstScalar;

  // The bias vector data, if not null.
  const AccumScalar* bias() const { return storage_.bias; }
  void set_bias(const AccumScalar* ptr) { storage_.bias = ptr; }
  // Only for non-floating-point cases. The fixed-point part (i.e. the mantissa)
  // of the multiplier by which accumulators are multiplied before being casted
  // to the destination type.
  AccumScalar multiplier_fixedpoint() const {
    return storage_.multiplier_fixedpoint;
  }
  void set_multiplier_fixedpoint(const AccumScalar value) {
    storage_.multiplier_fixedpoint = value;
  }
  // Only for non-floating-point cases. The exponent part of the aforementioned
  // multiplier.
  int multiplier_exponent() const { return storage_.multiplier_exponent; }
  void set_multiplier_exponent(const int value) {
    storage_.multiplier_exponent = value;
  }
  // Per-channel variant of multiplier_fixedpoint. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_fixedpoint.
  const AccumScalar* multiplier_fixedpoint_perchannel() const {
    return storage_.multiplier_fixedpoint_perchannel;
  }
  void set_multiplier_fixedpoint_perchannel(const AccumScalar* ptr) {
    storage_.multiplier_fixedpoint_perchannel = ptr;
  }
  // Per-channel variant of multiplier_exponent. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_exponent.
  //
  // Either none or both of multiplier_exponent_perchannel and
  // multiplier_fixedpoint_perchannel must be nullptr.
  const int* multiplier_exponent_perchannel() const {
    return storage_.multiplier_exponent_perchannel;
  }
  void set_multiplier_exponent_perchannel(const int* ptr) {
    storage_.multiplier_exponent_perchannel = ptr;
  }
  // min clamp bound of destination values.
  DstScalar clamp_min() const { return storage_.clamp_min; }
  void set_clamp_min(const DstScalar value) { storage_.clamp_min = value; }
  // max clamp bound of destination values.
  DstScalar clamp_max() const { return storage_.clamp_max; }
  void set_clamp_max(const DstScalar value) { storage_.clamp_max = value; }
  // Designates which dimension is the 'channels', for per-channel features
  // such as bias-addition and per-channel quantization multipliers.
  ChannelDimension channel_dimension() const {
    return storage_.channel_dimension;
  }
  void set_channel_dimension(ChannelDimension value) {
    storage_.channel_dimension = value;
  }

 private:
  detail::MulParamsStorage<AccumScalar, DstScalar> storage_;
};

namespace detail {

// Floating-point case.
template <typename AccumScalar, typename DstScalar>
struct MulParamsStorage final {
  static_assert(std::is_floating_point<AccumScalar>::value, "");
  static_assert(std::is_floating_point<DstScalar>::value, "");
  static_assert(sizeof(DstScalar) <= sizeof(AccumScalar), "");

  const AccumScalar* bias = nullptr;
  DstScalar clamp_min = -std::numeric_limits<DstScalar>::infinity();
  DstScalar clamp_max = std::numeric_limits<DstScalar>::infinity();
  ChannelDimension channel_dimension = ChannelDimension::kRow;

  // Data members that are disabled in this case are left as `static constexpr`
  // so that one can write some generic code.
  static constexpr const AccumScalar* multiplier_fixedpoint_perchannel =
      nullptr;
  static constexpr const int* multiplier_exponent_perchannel = nullptr;
  static constexpr AccumScalar multiplier_fixedpoint = 0;
  static constexpr int multiplier_exponent = 0;
};

// Specialization for the integer-quantized type, with down-quantization of
// int32 accumulators to a narrower destination scalar type.
template <typename DstScalar>
struct MulParamsStorage<std::int32_t, DstScalar> final {
  using AccumScalar = std::int32_t;
  static_assert(std::is_integral<DstScalar>::value, "");
  static_assert(sizeof(DstScalar) < sizeof(AccumScalar), "");

  const AccumScalar* bias = nullptr;
  const AccumScalar* multiplier_fixedpoint_perchannel = nullptr;
  const int* multiplier_exponent_perchannel = nullptr;
  AccumScalar multiplier_fixedpoint = 0;
  int multiplier_exponent = 0;
  DstScalar clamp_min = std::numeric_limits<DstScalar>::lowest();
  DstScalar clamp_max = std::numeric_limits<DstScalar>::max();
  ChannelDimension channel_dimension = ChannelDimension::kRow;
};

// Specialization used in the integer case when outputting raw int32
// accumulators, without down-quantization to a narrower destination scalar
// type. In this case, the feature of clamping destination values is not
// available.
template <>
struct MulParamsStorage<std::int32_t, std::int32_t> final {
  using AccumScalar = std::int32_t;
  using DstScalar = std::int32_t;

  const AccumScalar* bias = nullptr;
  ChannelDimension channel_dimension = ChannelDimension::kRow;

  // Data members that are disabled in this case are left as `static constexpr`
  // so that one can write some generic code.
  static constexpr const AccumScalar* multiplier_fixedpoint_perchannel =
      nullptr;
  static constexpr const int* multiplier_exponent_perchannel = nullptr;
  static constexpr AccumScalar multiplier_fixedpoint = 0;
  static constexpr int multiplier_exponent = 0;
  static constexpr DstScalar clamp_min =
      std::numeric_limits<DstScalar>::lowest();
  static constexpr DstScalar clamp_max = std::numeric_limits<DstScalar>::max();
};

}  // namespace detail

}  // namespace ruy

#endif  // RUY_RUY_MUL_PARAMS_H_
