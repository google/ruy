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

#ifndef RUY_RUY_KERNEL_X86_H_
#define RUY_RUY_KERNEL_X86_H_

#include <cstdint>

#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM_X86

RUY_INHERIT_KERNEL(Path::kStandardCpp, Path::kAvx2Fma)
RUY_INHERIT_KERNEL(Path::kStandardCpp, Path::kAvx)
RUY_INHERIT_KERNEL(Path::kAvx2Fma, Path::kAvx512)

void Kernel8bitAvx512(const KernelParams8bit<16, 16>& params);
void Kernel8bitAvx512SingleCol(const KernelParams8bit<16, 16>& params);

template <typename DstScalar>
struct Kernel<Path::kAvx512, std::int8_t, std::int8_t, std::int32_t, DstScalar> {
  static constexpr Path kPath = Path::kAvx512;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<std::int8_t>& lhs, const PMat<std::int8_t>& rhs,
           const MulParams<std::int32_t, DstScalar>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, mul_params, start_row, start_col, end_row,
                         end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      Kernel8bitAvx512SingleCol(params);
    } else {
      Kernel8bitAvx512(params);
    }
  }
};

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params);
void KernelFloatAvx512SingleCol(const KernelParamsFloat<16, 16>& param);

template <>
struct Kernel<Path::kAvx512, float, float, float, float> {
  static constexpr Path kPath = Path::kAvx512;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<float>& lhs, const PMat<float>& rhs,
           const MulParams<float, float>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, mul_params, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      KernelFloatAvx512SingleCol(params);
    } else {
      KernelFloatAvx512(params);
    }
  }
};

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params);
void Kernel8bitAvx2SingleCol(const KernelParams8bit<8, 8>& params);

template <typename DstScalar>
struct Kernel<Path::kAvx2Fma, std::int8_t, std::int8_t, std::int32_t,
              DstScalar> {
  static constexpr Path kPath = Path::kAvx2Fma;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<std::int8_t>& lhs, const PMat<std::int8_t>& rhs,
           const MulParams<std::int32_t, DstScalar>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, mul_params, start_row, start_col, end_row,
                         end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      Kernel8bitAvx2SingleCol(params);
    } else {
      Kernel8bitAvx2(params);
    }
  }
};

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params);
void KernelFloatAvx2SingleCol(const KernelParamsFloat<8, 8>& params);

template <>
struct Kernel<Path::kAvx2Fma, float, float, float, float> {
  static constexpr Path kPath = Path::kAvx2Fma;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<float>& lhs, const PMat<float>& rhs,
           const MulParams<float, float>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, mul_params, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      KernelFloatAvx2SingleCol(params);
    } else {
      KernelFloatAvx2(params);
    }
  }
};

void KernelFloatAvx(const KernelParamsFloat<8, 8>& params);
void KernelFloatAvxSingleCol(const KernelParamsFloat<8, 8>& params);

template <>
struct Kernel<Path::kAvx, float, float, float, float> {
  static constexpr Path kPath = Path::kAvx;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<float>& lhs, const PMat<float>& rhs,
           const MulParams<float, float>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, mul_params, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      KernelFloatAvxSingleCol(params);
    } else {
      KernelFloatAvx(params);
    }
  }
};
#endif  // RUY_PLATFORM_X86
}  // namespace ruy

#if ((RUY_PLATFORM_AVX || RUY_PLATFORM_AVX2_FMA) && RUY_OPT(ASM))

#include <immintrin.h>  // IWYU pragma: keep

namespace ruy {
namespace {
namespace intrin_utils {

// Defined as a template so clang won't detect it as an uneeded
// definition.
template <Path path>
inline float mm256_get1_ps(const __m256 a, int i) {
  __m256i ai = _mm256_castps_si256(a);
  int float_val_as_int;
  switch (i) {
    case 0:
      float_val_as_int = _mm256_extract_epi32(ai, 0);
      break;
    case 1:
      float_val_as_int = _mm256_extract_epi32(ai, 1);
      break;
    case 2:
      float_val_as_int = _mm256_extract_epi32(ai, 2);
      break;
    case 3:
      float_val_as_int = _mm256_extract_epi32(ai, 3);
      break;
    case 4:
      float_val_as_int = _mm256_extract_epi32(ai, 4);
      break;
    case 5:
      float_val_as_int = _mm256_extract_epi32(ai, 5);
      break;
    case 6:
      float_val_as_int = _mm256_extract_epi32(ai, 6);
      break;
    case 7:
      float_val_as_int = _mm256_extract_epi32(ai, 7);
      break;
    default:
      RUY_DCHECK_LT(i, 8);
      return .0f;
  }
  float float_val;
  std::memcpy(&float_val, &float_val_as_int, sizeof(float_val));
  return float_val;
}

// Defined as a template so clang won't detect it as an uneeded
// definition.
template <Path path>
inline void mm256_n_storeu_ps(float* dst, int residual_rows, const __m256 v) {
  for (int i = 0; i < residual_rows; ++i) {
    dst[i] = intrin_utils::mm256_get1_ps<path>(v, i);
  }
}

template <Path path>
inline __m256 MulAdd(const __m256&, const __m256&, const __m256&) {
  // Specializations added for AVX and AVX2FMA paths in their respective kernel
  // files.
  RUY_DCHECK(false);
}
}  // namespace intrin_utils
}  // namespace

template <Path path>
inline void KernelFloatAvxCommon(const KernelParamsFloat<8, 8>& params) {
  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;
  const std::int64_t dst_stride = params.dst_stride >> 2;
  const std::int64_t rhs_stride = params.rhs_stride >> 2;
  //
  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  // AVX2 float block size = 8.
  const int end_row = std::min(params.dst_rows, params.last_row + 8);
  const int end_col = std::min(params.dst_cols, params.last_col + 8);
  //
  const float* adj_rhs_col_ptr =
      params.rhs_base_ptr - params.start_col * rhs_stride;
  float* adj_dst_col_ptr =
      params.dst_base_ptr - params.start_col * dst_stride - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_ptr = params.bias;

  const __m256 clamp_max_v = _mm256_set1_ps(params.clamp_max);
  const __m256 clamp_min_v = _mm256_set1_ps(params.clamp_min);
  const bool channel_dimension_is_col =
      params.flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL;

  int col = params.start_col;
  // Loop through cols by float block size, leaving incomplete remainder
  for (; col <= end_col - 8; col += 8) {
    __m256 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;

    for (int row = params.start_row; row < end_row; row += 8) {
      const int residual_rows = std::min(end_row - row, 8);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;

      // Initialize with bias.
      if (channel_dimension_is_col) {
        const float* bias_elem_ptr = bias_ptr + col * bias_ptr_block_increment;
        for (int j = 0; j < 8; ++j) {
          accum_data_v[j] = _mm256_broadcast_ss(bias_elem_ptr + j);
        }
      } else {
        const float* bias_elem_ptr = bias_ptr + row * bias_ptr_block_increment;
        const __m256 initial_accum_data = _mm256_loadu_ps(bias_elem_ptr);

        for (int j = 0; j < 8; ++j) {
          accum_data_v[j] = initial_accum_data;
        }
      }

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
        // In this version RHS values are loaded individually rather than first
        // loading together and then extract with broadcasting. This is because
        // AVX flavours and instrinsics and compilers in combination do not
        // handle this pattern of extraction very well.
        const float* rhs_data = rhs_ptr;

        for (int j = 0; j < 8; ++j) {
          const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[j]);
          accum_data_v[j] = intrin_utils::MulAdd<path>(
              lhs_data, dup_rhs_element_j, accum_data_v[j]);
        }
        lhs_ptr += 8;
        rhs_ptr += 8;
      }

      if (residual_rows == 8) {
        for (int j = 0; j < 8; ++j) {
          float* block_ptr = dst_ptr + j * dst_stride;
          accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
          _mm256_storeu_ps(block_ptr, accum_data_v[j]);
        }
      } else {
        for (int j = 0; j < 8; ++j) {
          float* block_ptr = dst_ptr + j * dst_stride;
          accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
          intrin_utils::mm256_n_storeu_ps<path>(block_ptr, residual_rows,
                                                accum_data_v[j]);
        }
      }
    }  // End row-block loop.
  }    // End col-block loop.

  if (col < end_col) {
    // Remaining cols in [0, float block size).
    RUY_DCHECK_GE(end_col - col, 0);
    RUY_DCHECK_LT(end_col - col, 8);

    __m256 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;
    const int residual_cols = std::min(end_col - col, 8);

    for (int row = params.start_row; row < end_row; row += 8) {
      const int residual_rows = std::min(end_row - row, 8);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;

      // Initialize with bias.
      if (channel_dimension_is_col) {
        const float* bias_elem_ptr = bias_ptr + col * bias_ptr_block_increment;
        for (int j = 0; j < 8; ++j) {
          accum_data_v[j] = _mm256_broadcast_ss(bias_elem_ptr + j);
        }
      } else {
        const float* bias_elem_ptr = bias_ptr + row * bias_ptr_block_increment;
        const __m256 initial_accum_data = _mm256_loadu_ps(bias_elem_ptr);

        for (int j = 0; j < 8; ++j) {
          accum_data_v[j] = initial_accum_data;
        }
      }

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
        const float* rhs_data = rhs_ptr;

        for (int j = 0; j < 8; ++j) {
          const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[j]);
          accum_data_v[j] = intrin_utils::MulAdd<path>(
              lhs_data, dup_rhs_element_j, accum_data_v[j]);
        }
        lhs_ptr += 8;
        rhs_ptr += 8;
      }

      for (int j = 0; j < residual_cols; ++j) {
        float* block_ptr = dst_ptr + j * dst_stride;
        accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
        accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
        intrin_utils::mm256_n_storeu_ps<path>(block_ptr, residual_rows,
                                              accum_data_v[j]);
      }
    }  // End row-block loop.
  }    // End col-block terminal conditional.
}

template <Path path>
inline void KernelFloatAvxCommonSingleCol(
    const KernelParamsFloat<8, 8>& params) {
  RUY_DCHECK_EQ(params.dst_cols, 1);
  RUY_DCHECK_EQ(params.last_col, 0);
  RUY_DCHECK_EQ(params.start_col, 0);

  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;
  //
  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  // AVX2 float block size = 8.
  const int end_row = std::min(params.dst_rows, params.last_row + 8);

  float* adj_dst_col_ptr = params.dst_base_ptr - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_col_ptr = params.bias;

  const __m256 clamp_max_v = _mm256_set1_ps(params.clamp_max);
  const __m256 clamp_min_v = _mm256_set1_ps(params.clamp_min);

  __m256 accum_data_v;

  const float* rhs_col_ptr = params.rhs_base_ptr;
  float* dst_col_ptr = adj_dst_col_ptr;

  int row = params.start_row;
  for (; row <= end_row - 8; row += 8) {
    const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
    float* dst_ptr = dst_col_ptr + row;
    const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

    // Initialize with bias.
    accum_data_v = _mm256_loadu_ps(bias_ptr);

    const float* lhs_ptr = lhs_col_ptr;
    const float* rhs_ptr = rhs_col_ptr;
    int d = 0;
    for (; d <= params.depth - 4; d += 4) {
      const __m256 lhs_data_0 = _mm256_loadu_ps(lhs_ptr);
      const __m256 dup_rhs_element_0 = _mm256_set1_ps(rhs_ptr[0]);
      accum_data_v = intrin_utils::MulAdd<path>(lhs_data_0, dup_rhs_element_0,
                                                accum_data_v);
      const __m256 dup_rhs_element_1 = _mm256_set1_ps(rhs_ptr[8]);
      const __m256 lhs_data_1 = _mm256_loadu_ps(lhs_ptr + 8);
      accum_data_v = intrin_utils::MulAdd<path>(lhs_data_1, dup_rhs_element_1,
                                                accum_data_v);

      const __m256 lhs_data_2 = _mm256_loadu_ps(lhs_ptr + 16);
      const __m256 dup_rhs_element_2 = _mm256_set1_ps(rhs_ptr[16]);
      accum_data_v = intrin_utils::MulAdd<path>(lhs_data_2, dup_rhs_element_2,
                                                accum_data_v);
      const __m256 dup_rhs_element_3 = _mm256_set1_ps(rhs_ptr[24]);
      const __m256 lhs_data_3 = _mm256_loadu_ps(lhs_ptr + 24);
      accum_data_v = intrin_utils::MulAdd<path>(lhs_data_3, dup_rhs_element_3,
                                                accum_data_v);
      lhs_ptr += 32;  // Loaded 8 * 4 floats.
      rhs_ptr += 32;
    }
    for (; d < params.depth; ++d) {
      const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
      const float* rhs_data = rhs_ptr;

      const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[0]);
      accum_data_v =
          intrin_utils::MulAdd<path>(lhs_data, dup_rhs_element_j, accum_data_v);
      lhs_ptr += 8;
      rhs_ptr += 8;
    }

    accum_data_v = _mm256_min_ps(accum_data_v, clamp_max_v);
    accum_data_v = _mm256_max_ps(accum_data_v, clamp_min_v);
    _mm256_storeu_ps(dst_ptr, accum_data_v);
  }  // End row-block loop.

  if (row < end_row) {
    const int residual_rows = end_row - row;
    RUY_CHECK_GE(residual_rows, 1);
    RUY_CHECK_LT(residual_rows, 8);

    const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
    float* dst_ptr = dst_col_ptr + row;
    const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

    // Initialize with bias.
    accum_data_v = _mm256_loadu_ps(bias_ptr);

    const float* lhs_ptr = lhs_col_ptr;
    const float* rhs_ptr = rhs_col_ptr;
    for (int d = 0; d < params.depth; ++d) {
      const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
      const float* rhs_data = rhs_ptr;

      const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[0]);
      accum_data_v =
          intrin_utils::MulAdd<path>(lhs_data, dup_rhs_element_j, accum_data_v);
      lhs_ptr += 8;
      rhs_ptr += 8;
    }

    accum_data_v = _mm256_min_ps(accum_data_v, clamp_max_v);
    accum_data_v = _mm256_max_ps(accum_data_v, clamp_min_v);
    intrin_utils::mm256_n_storeu_ps<path>(dst_ptr, residual_rows, accum_data_v);
  }  // End handling of residual rows.
}
}  // namespace ruy
#endif  //  (RUY_PLATFORM_AVX || RUY_PLATFORM_AVX2_FMA) && RUY_OPT(ASM)

#endif  // RUY_RUY_KERNEL_X86_H_
