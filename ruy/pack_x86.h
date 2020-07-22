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

#ifndef RUY_RUY_PACK_X86_H_
#define RUY_RUY_PACK_X86_H_

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ruy/check_macros.h"
#include "ruy/mat.h"
#include "ruy/opt_set.h"
#include "ruy/pack_common.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM_X86

RUY_INHERIT_PACK(Path::kStandardCpp, Path::kAvx2Fma)
RUY_INHERIT_PACK(Path::kAvx2Fma, Path::kAvx)
RUY_INHERIT_PACK(Path::kAvx2Fma, Path::kAvx512)

RUY_USE_MEMCPY_ROWMAJOR_FLOAT_PACK(Path::kAvx2Fma, 8)
RUY_USE_MEMCPY_ROWMAJOR_FLOAT_PACK(Path::kAvx512, 16)

template <>
struct PackedTypeImpl<Path::kAvx2Fma, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kAvx512, std::uint8_t> {
  using Type = std::int8_t;
};

// Note that source and zero buffers can be uint8 type, but in the packing
// function are reinterpreted as int8, and are XOR-ed with input_xor.
void Pack8bitColMajorForAvx2(const std::int8_t* src_ptr, std::int8_t input_xor,
                             const std::int8_t* zerobuf, int src_stride,
                             int remaining_src_cols, int src_rows,
                             std::int8_t* packed_ptr, std::int32_t* sums_ptr);

template <typename Scalar>
struct PackImpl<Path::kAvx2Fma, FixedKernelLayout<Order::kColMajor, 4, 8>,
                Scalar, std::int8_t, std::int32_t, Order::kColMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  using Layout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  static constexpr std::int8_t kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    profiler::ScopeLabel label("Pack (AVX2 8-bit)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[Layout::kCols * Layout::kRows];
    memset(zerobuf, packed_matrix->zero_point ^ kInputXor,
           Layout::kCols * Layout::kRows * sizeof(Scalar));
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      Pack8bitColMajorForAvx2(
          reinterpret_cast<const std::int8_t*>(src_ptr), kInputXor,
          reinterpret_cast<const std::int8_t*>(zerobuf), src_stride,
          remaining_src_cols, src_matrix.layout.rows, packed_ptr, sums_ptr);
    }
  }
};

void PackFloatColMajorForAvx(const float* src_ptr, const float* zerobuf,
                             int src_stride, int remaining_src_cols,
                             int src_rows, float* packed_ptr);

template <>
struct PackImpl<Path::kAvx, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
                float, float, Order::kColMajor> {
  using Layout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  static void Run(Tuning, const Mat<float>& src_matrix,
                  PMat<float>* packed_matrix, int start_col, int end_col) {
    profiler::ScopeLabel label("Pack (AVX float)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    const float zerobuf[Layout::kCols] = {
        0.0f};  // Remainder default inits to 0.0f.
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      float* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      PackFloatColMajorForAvx(src_ptr, zerobuf, src_stride, remaining_src_cols,
                              src_matrix.layout.rows, packed_ptr);
    }
  }
};

void PackFloatColMajorForAvx2(const float* src_ptr, const float* zerobuf,
                              int src_stride, int remaining_src_cols,
                              int src_rows, float* packed_ptr);

template <>
struct PackImpl<Path::kAvx2Fma, FixedKernelLayout<Order::kRowMajor, 1, 8>,
                float, float, float, Order::kColMajor> {
  using Layout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  static void Run(Tuning, const Mat<float>& src_matrix,
                  PMat<float>* packed_matrix, int start_col, int end_col) {
    profiler::ScopeLabel label("Pack (AVX2 float)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    const float zerobuf[Layout::kCols] = {
        0.0f};  // Remainder default inits to 0.0f.
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      float* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      PackFloatColMajorForAvx2(src_ptr, zerobuf, src_stride, remaining_src_cols,
                               src_matrix.layout.rows, packed_ptr);
    }
  }
};

// Note that source and zero buffers can be uint8 type, but in the packing
// function are reinterpreted as int8, and are XOR-ed with input_xor.
void Pack8bitColMajorForAvx512(const std::int8_t* src_ptr,
                               std::int8_t input_xor,
                               const std::int8_t* zerobuf, int src_stride,
                               int remaining_src_cols, int src_rows,
                               std::int8_t* packed_ptr, std::int32_t* sums_ptr);

template <typename Scalar>
struct PackImpl<Path::kAvx512, FixedKernelLayout<Order::kColMajor, 4, 16>,
                Scalar, std::int8_t, std::int32_t, Order::kColMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  using Layout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  static constexpr int kHalfLayoutCols =
      8;  // Half the number of cols in a block.
  static constexpr std::int8_t kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    profiler::ScopeLabel label("Pack (AVX-512 8-bit)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    RUY_DCHECK_EQ(kHalfLayoutCols * 2, Layout::kCols);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[kHalfLayoutCols * Layout::kRows];
    memset(zerobuf, packed_matrix->zero_point ^ kInputXor,
           kHalfLayoutCols * Layout::kRows * sizeof(Scalar));
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      Pack8bitColMajorForAvx512(
          reinterpret_cast<const std::int8_t*>(src_ptr), kInputXor,
          reinterpret_cast<const std::int8_t*>(zerobuf), src_stride,
          remaining_src_cols, src_matrix.layout.rows, packed_ptr, sums_ptr);
    }
  }
};

void PackFloatColMajorForAvx512(const float* src_ptr, const float* zerobuf,
                                int src_stride, int remaining_src_cols,
                                int src_rows, float* packed_ptr);

template <>
struct PackImpl<Path::kAvx512, FixedKernelLayout<Order::kRowMajor, 1, 16>,
                float, float, float, Order::kColMajor> {
  static void Run(Tuning, const Mat<float>& src_matrix,
                  PMat<float>* packed_matrix, int start_col, int end_col) {
    profiler::ScopeLabel label("Pack (AVX-512 float)");
    using Layout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    const float zerobuf[Layout::kCols] = {
        0.0f};  // Remainder default inits to 0.0f.
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      float* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      PackFloatColMajorForAvx512(src_ptr, zerobuf, src_stride,
                                 remaining_src_cols, src_matrix.layout.rows,
                                 packed_ptr);
    }
  }
};
#endif  // RUY_PLATFORM_X86

}  // namespace ruy

#endif  // RUY_RUY_PACK_X86_H_
