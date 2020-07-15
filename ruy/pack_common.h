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

#ifndef RUY_RUY_PACK_COMMON_H_
#define RUY_RUY_PACK_COMMON_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ruy/check_macros.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/tune.h"

namespace ruy {

template <typename Scalar>
Scalar SymmetricZeroPoint() {
  if (std::is_floating_point<Scalar>::value) {
    return 0;
  }
  if (std::is_signed<Scalar>::value) {
    return 0;
  }
  return std::numeric_limits<Scalar>::max() / 2 + 1;
}

template <Path ThePath, typename Scalar>
struct PackedTypeImpl {
  using Type = Scalar;
};

template <Path ThePath, typename Scalar>
using PackedType = typename PackedTypeImpl<ThePath, Scalar>::Type;

template <typename PackedScalar, typename Scalar>
PackedScalar Pack(Scalar x) {
  return x - SymmetricZeroPoint<Scalar>() + SymmetricZeroPoint<PackedScalar>();
}

template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar, typename SumsType, Order SrcOrder>
struct PackImpl;

#define RUY_INHERIT_PACK(PARENT, CHILD)                                     \
  template <typename FixedKernelLayout, typename Scalar,                    \
            typename PackedScalar, typename SumsType, Order SrcOrder>       \
  struct PackImpl<CHILD, FixedKernelLayout, Scalar, PackedScalar, SumsType, \
                  SrcOrder> : PackImpl<PARENT, FixedKernelLayout, Scalar,   \
                                       PackedScalar, SumsType, SrcOrder> {};

}  // namespace ruy

#endif  // RUY_RUY_PACK_COMMON_H_
