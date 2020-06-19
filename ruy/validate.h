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

// Front-end validation code, see the Validate function.

#ifndef RUY_RUY_VALIDATE_H_
#define RUY_RUY_VALIDATE_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ruy/check_macros.h"
#include "ruy/common.h"
#include "ruy/mat.h"
#include "ruy/mul_params.h"
#include "ruy/side_pair.h"

namespace ruy {
namespace detail {

template <typename Scalar>
void CheckZeroPoint(Scalar zero_point) {
  if (std::is_floating_point<Scalar>::value) {
    RUY_DCHECK(!zero_point);
  }
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void ValidateZeroPoints(LhsScalar lhs_zero_point, RhsScalar rhs_zero_point,
                        DstScalar dst_zero_point) {
  CheckZeroPoint(lhs_zero_point);
  CheckZeroPoint(rhs_zero_point);
  CheckZeroPoint(dst_zero_point);

  // Guard against the case when both LHS and RHS zero_point's are equal to
  // the minimum representable value. In that case, padding with zero_point
  // values will generate the bad case for fast int8 kernels on NEON
  // (pre-dotprod) which attempt to multiply-accumulate two pairs of int8
  // into a int16:  this is safe except in the bad case -128*-128 + -128*-128.
  // See b/131609283. This only affects the kNeon path but we ban this for all
  // paths in order for ruy to have the same supported parameter space
  // on all paths.
  RUY_DCHECK(lhs_zero_point != std::numeric_limits<LhsScalar>::lowest() ||
             rhs_zero_point != std::numeric_limits<RhsScalar>::lowest());
}

template <typename MulParamsType, typename DstScalar>
void ValidateRawAccumulatorsDst(const MulParamsType& mul_params,
                                DstScalar dst_zero_point) {
  static_assert(
      std::is_same<typename MulParamsType::DstScalar, DstScalar>::value, "");
  if (!std::is_same<typename MulParamsType::DstScalar, std::int32_t>::value)
    return;

  // If user is looking for the raw accumulator, zero_point and all the other
  // dequantize fields don't make sense and should not be set.
  RUY_DCHECK_EQ(dst_zero_point, 0);
  RUY_DCHECK_EQ(mul_params.clamp_max(),
                std::numeric_limits<std::int32_t>::max());
  RUY_DCHECK_EQ(mul_params.clamp_min(),
                std::numeric_limits<std::int32_t>::min());
  RUY_DCHECK_EQ(mul_params.multiplier_fixedpoint(), 0);
  RUY_DCHECK_EQ(mul_params.multiplier_exponent(), 0);
  RUY_DCHECK_EQ(mul_params.multiplier_fixedpoint_perchannel(), nullptr);
  RUY_DCHECK_EQ(mul_params.multiplier_exponent_perchannel(), nullptr);
}

}  // namespace detail

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
void Validate(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
              const Mat<DstScalar>& dst, const MulParamsType& mul_params) {
  detail::ValidateZeroPoints(lhs.zero_point, rhs.zero_point, dst.zero_point);
  detail::ValidateRawAccumulatorsDst<MulParamsType>(mul_params, dst.zero_point);
}

}  // namespace ruy

#endif  // RUY_RUY_VALIDATE_H_
