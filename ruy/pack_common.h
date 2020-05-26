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

#include "ruy/check_macros.h"
#include "ruy/common.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/tune.h"

namespace ruy {

template <Path ThePath, typename Scalar>
struct PackedTypeImpl {
  using Type = Scalar;
};

#if RUY_PLATFORM_NEON_32
struct PackParams8bit {
  const void* src_ptr0;
  const void* src_ptr1;
  const void* src_ptr2;
  const void* src_ptr3;
  const std::int32_t* sums_ptr;
  const std::int8_t* packed_ptr;
  int src_inc0;
  int src_inc1;
  int src_inc2;
  int src_inc3;
  int src_rows;
  int src_zero_point;
  int input_xor;
};

inline void MakePackParams8bit(const void* src_ptr0, const void* src_ptr1,
                               const void* src_ptr2, const void* src_ptr3,
                               const std::int32_t* sums_ptr,
                               const std::int8_t* packed_ptr, int src_inc0,
                               int src_inc1, int src_inc2, int src_inc3,
                               int src_rows, int src_zero_point, int input_xor,
                               PackParams8bit* params) {
  params->src_ptr0 = src_ptr0;
  params->src_ptr1 = src_ptr1;
  params->src_ptr2 = src_ptr2;
  params->src_ptr3 = src_ptr3;
  params->sums_ptr = sums_ptr;
  params->packed_ptr = packed_ptr;
  params->src_inc0 = src_inc0;
  params->src_inc1 = src_inc1;
  params->src_inc2 = src_inc2;
  params->src_inc3 = src_inc3;
  params->src_rows = src_rows;
  params->src_zero_point = src_zero_point;
  params->input_xor = input_xor;
}
#endif

#if RUY_PLATFORM_NEON
template <>
struct PackedTypeImpl<Path::kNeon, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kNeonDotprod, std::uint8_t> {
  using Type = std::int8_t;
};
#elif RUY_PLATFORM_X86
template <>
struct PackedTypeImpl<Path::kSse42, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kAvx2, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kAvx512, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kAvxVnni, std::uint8_t> {
  using Type = std::int8_t;
};
#endif

template <Path ThePath, typename Scalar>
using PackedType = typename PackedTypeImpl<ThePath, Scalar>::Type;

template <typename PackedScalar, typename Scalar>
PackedScalar Pack(Scalar x) {
  return x - SymmetricZeroPoint<Scalar>() + SymmetricZeroPoint<PackedScalar>();
}

template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar, typename SumsType>
struct PackImpl {};

#define RUY_INHERIT_PACK(PARENT, CHILD)                                       \
  template <typename FixedKernelLayout, typename Scalar,                      \
            typename PackedScalar, typename SumsType>                         \
  struct PackImpl<CHILD, FixedKernelLayout, Scalar, PackedScalar, SumsType>   \
      : PackImpl<PARENT, FixedKernelLayout, Scalar, PackedScalar, SumsType> { \
  };

template <typename FixedKernelLayout, typename Scalar, typename PackedScalar,
          typename SumsType>
struct PackImpl<Path::kStandardCpp, FixedKernelLayout, Scalar, PackedScalar,
                SumsType> {
  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<PackedScalar>* packed_matrix, int start_col,
                  int end_col) {
    profiler::ScopeLabel label("Pack (generic)");
    RUY_DCHECK_EQ((end_col - start_col) % FixedKernelLayout::kCols, 0);
    SumsType* sums = packed_matrix->sums;
    for (int col = start_col; col < end_col; col++) {
      SumsType accum = 0;
      for (int row = 0; row < packed_matrix->layout.rows; row++) {
        PackedScalar packed_val;
        if (col < src_matrix.layout.cols && row < src_matrix.layout.rows) {
          packed_val = Pack<PackedScalar>(Element(src_matrix, row, col));
        } else {
          packed_val = packed_matrix->zero_point;
        }
        accum += packed_val;
        *ElementPtr(packed_matrix, row, col) = packed_val;
      }
      if (sums) {
        sums[col] = accum;
      }
    }
  }
};

#if RUY_PLATFORM_NEON
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kNeon)
RUY_INHERIT_PACK(Path::kNeon, Path::kNeonDotprod)
#elif RUY_PLATFORM_X86
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kSse42)
RUY_INHERIT_PACK(Path::kSse42, Path::kAvx2)
RUY_INHERIT_PACK(Path::kAvx2, Path::kAvx512)
RUY_INHERIT_PACK(Path::kAvx512, Path::kAvxVnni)
#endif

// Main entry point for packing.
template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar>
void RunPack(Tuning tuning, const EMat& src_matrix, PEMat* packed_matrix,
             int start_col, int end_col) {
  using SumsType = typename PMat<PackedScalar>::SumsType;
  Mat<Scalar> src = UneraseType<Scalar>(src_matrix);
  PMat<PackedScalar> packed = UneraseType<PackedScalar>(*packed_matrix);
  PackImpl<ThePath, FixedKernelLayout, Scalar, PackedScalar, SumsType>::Run(
      tuning, src, &packed, start_col, end_col);
}

}  // namespace ruy

#endif  // RUY_RUY_PACK_COMMON_H_
