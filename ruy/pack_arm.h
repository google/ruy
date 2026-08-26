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

#ifndef RUY_RUY_PACK_ARM_H_
#define RUY_RUY_PACK_ARM_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ruy/asm_helpers.h"
#include "ruy/check_macros.h"
#include "ruy/mat.h"
#include "ruy/opt_set.h"
#include "ruy/pack_common.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM_NEON
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kNeon)
RUY_INHERIT_PACK(Path::kNeon, Path::kNeonDotprod)

RUY_USE_MEMCPY_ROWMAJOR_FLOAT_PACK(Path::kNeon, 8)
#if RUY_PLATFORM_NEON_32
RUY_USE_MEMCPY_ROWMAJOR_FLOAT_PACK(Path::kNeon, 4)
#endif

template <>
struct PackedTypeImpl<Path::kNeon, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kNeonDotprod, std::uint8_t> {
  using Type = std::int8_t;
};
#endif

#if RUY_PLATFORM_NEON
void Pack8bitRowMajorForNeon(const std::uint8_t* src_ptr, int src_stride,
                             int src_rows, int src_cols, int block_row,
                             int start_col, int end_col,
                             std::int8_t* packed_ptr, int packed_stride,
                             int packed_zero_point, std::int32_t* sums_ptr,
                             int input_xor, int kernel_cols);
#endif

#if (RUY_PLATFORM_NEON_64 && RUY_OPT(ASM)) && \
    (!defined(_MSC_VER) || defined(_M_ARM64))

void Pack8bitColMajorForNeon(const void* src_ptr0, const void* src_ptr1,
                             const void* src_ptr2, const void* src_ptr3,
                             int src_inc0, int src_inc1, int src_inc2,
                             int src_inc3, int src_rows, int src_zero_point,
                             std::int8_t* packed_ptr, std::int32_t* sums_ptr,
                             int input_xor);
void Pack8bitColMajorForNeonA55ish(const void* src_ptr0, const void* src_ptr1,
                                   const void* src_ptr2, const void* src_ptr3,
                                   int src_inc0, int src_inc1, int src_inc2,
                                   int src_inc3, int src_rows,
                                   int src_zero_point, std::int8_t* packed_ptr,
                                   std::int32_t* sums_ptr, int input_xor);
void Pack8bitColMajorForNeonDotprod(const void* src_ptr0, const void* src_ptr1,
                                    const void* src_ptr2, const void* src_ptr3,
                                    int src_inc0, int src_inc1, int src_inc2,
                                    int src_inc3, int src_rows,
                                    int src_zero_point, std::int8_t* packed_ptr,
                                    std::int32_t* sums_ptr, int input_xor);
void Pack8bitColMajorForNeonDotprodA55ish(
    const void* src_ptr0, const void* src_ptr1, const void* src_ptr2,
    const void* src_ptr3, int src_inc0, int src_inc1, int src_inc2,
    int src_inc3, int src_rows, int src_zero_point, std::int8_t* packed_ptr,
    std::int32_t* sums_ptr, int input_xor);
void Pack8bitRowMajorForNeonDotprod(const void* src_ptr0, const void* src_ptr1,
                                    const void* src_ptr2, const void* src_ptr3,
                                    int src_inc0, int src_inc1, int src_inc2,
                                    int src_inc3, int src_cols,
                                    int src_zero_point, std::int8_t* packed_ptr,
                                    int packed_stride, std::int32_t* sums_ptr,
                                    int input_xor);
#elif RUY_PLATFORM_NEON_32 && RUY_OPT(ASM) && !defined(_MSC_VER)
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

void Pack8bitColMajorForNeon4Cols(const PackParams8bit& params);
void Pack8bitColMajorForNeon2Cols(const PackParams8bit& params);

#endif  // (RUY_PLATFORM_NEON_32 && RUY_OPT(ASM) && !defined(_MSC_VER)

#if (RUY_PLATFORM_NEON_32 || RUY_PLATFORM_NEON_64) && RUY_OPT(ASM) && \
    (!defined(_MSC_VER) || defined(_M_ARM64))

template <typename Scalar>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kColMajor, 16, 4>, Scalar,
                std::int8_t, std::int32_t, Order::kColMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      int src_inc0 = 16;
      int src_inc1 = 16;
      int src_inc2 = 16;
      int src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
#if RUY_PLATFORM_NEON_64
      if (__builtin_expect(tuning == Tuning::kA55ish, true)) {
        Pack8bitColMajorForNeonA55ish(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, sums_ptr, kInputXor);
      } else {
        Pack8bitColMajorForNeon(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                src_inc0, src_inc1, src_inc2, src_inc3,
                                src_matrix.layout.rows, src_matrix.zero_point,
                                packed_ptr, sums_ptr, kInputXor);
      }
#else
      (void)tuning;
      // We have a more limited set of general purpose registers in ARMv7, so
      // we use the "params" struct technique from the kernel code to save
      // registers.
      PackParams8bit params;
      MakePackParams8bit(src_ptr0, src_ptr1, src_ptr2, src_ptr3, sums_ptr,
                         packed_ptr, src_inc0, src_inc1, src_inc2, src_inc3,
                         src_matrix.layout.rows, src_matrix.zero_point,
                         kInputXor, &params);
      Pack8bitColMajorForNeon4Cols(params);
#endif  // RUY_PLATFORM_NEON_64
    }
  }
};

#endif  // (RUY_PLATFORM_NEON_32 || RUY_PLATFORM_NEON_64) &&
        // RUY_OPT(ASM)

#if defined(_MSC_VER) && defined(_M_ARM64)

// Pack int16 source, col-major, with running int32 column sums.
// 4 columns packed in parallel; each step: 8 int16 elements per col (=16 bytes).
// Output is identical int16 layout (no conversion), just copied with zero-pad.
template <>
struct PackImpl<Path::kNeon,
                FixedKernelLayout<Order::kColMajor, 8, 4>,
                std::int16_t, std::int16_t, std::int32_t,
                Order::kColMajor> {
  static void Run(Tuning, const Mat<std::int16_t>& src_matrix,
                  PMat<std::int16_t>* packed_matrix, int start_col,
                  int end_col) {
    profiler::ScopeLabel label("Pack (int16 kNeon ColMajor)");
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);

    const int src_stride = src_matrix.layout.stride;
    const int packed_stride = packed_matrix->layout.stride;
    const int src_rows = src_matrix.layout.rows;
    const int packed_rows = packed_matrix->layout.rows;
    std::int32_t* sums = packed_matrix->sums;
    const std::int16_t zero_pt = src_matrix.zero_point;

    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      // Pointers to each of the 4 source columns (may be clamped to a zero buf)
      std::int16_t zerobuf[8];
      for (int i = 0; i < 8; i++) zerobuf[i] = zero_pt;

      const std::int16_t* src0 = src_matrix.data.get() + src_stride * block_col;
      const std::int16_t* src1 = src0 + src_stride;
      const std::int16_t* src2 = src1 + src_stride;
      const std::int16_t* src3 = src2 + src_stride;
      int inc0 = 8, inc1 = 8, inc2 = 8, inc3 = 8;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols)     { src0 = zerobuf; inc0 = 0; }
        if (block_col >= src_matrix.layout.cols - 1) { src1 = zerobuf; inc1 = 0; }
        if (block_col >= src_matrix.layout.cols - 2) { src2 = zerobuf; inc2 = 0; }
        if (block_col >= src_matrix.layout.cols - 3) { src3 = zerobuf; inc3 = 0; }
      }

      std::int16_t* packed_ptr =
          packed_matrix->data + packed_stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;

      int32x4_t sum_vec = vdupq_n_s32(0);  // accumulates sums for cols 0..3

      int r = 0;
      for (; r + 8 <= src_rows; r += 8) {
        // Load 8 int16 from each of the 4 columns
        int16x8_t v0 = vld1q_s16(src0); src0 += inc0;
        int16x8_t v1 = vld1q_s16(src1); src1 += inc1;
        int16x8_t v2 = vld1q_s16(src2); src2 += inc2;
        int16x8_t v3 = vld1q_s16(src3); src3 += inc3;
        // Store interleaved: col0[0..7], col1[0..7], col2[0..7], col3[0..7]
        vst1q_s16(packed_ptr,      v0);
        vst1q_s16(packed_ptr + 8,  v1);
        vst1q_s16(packed_ptr + 16, v2);
        vst1q_s16(packed_ptr + 24, v3);
        packed_ptr += 32;
        // Accumulate column sums: pairwise add int16→int32, then accumulate
        // into a 4-element int32 vector (one per column).
        int32x4_t s0 = vpaddlq_s16(v0);
        int32x4_t s1 = vpaddlq_s16(v1);
        int32x4_t s2 = vpaddlq_s16(v2);
        int32x4_t s3 = vpaddlq_s16(v3);
        // Reduce each to a scalar via horizontal add
        int32x2_t r0 = vadd_s32(vget_low_s32(s0), vget_high_s32(s0));
        int32x2_t r1 = vadd_s32(vget_low_s32(s1), vget_high_s32(s1));
        int32x2_t r2 = vadd_s32(vget_low_s32(s2), vget_high_s32(s2));
        int32x2_t r3 = vadd_s32(vget_low_s32(s3), vget_high_s32(s3));
        // Combine to a 4-element int32 vector and accumulate
        int32x2_t lo = vpadd_s32(r0, r1);  // [sum0, sum1]
        int32x2_t hi = vpadd_s32(r2, r3);  // [sum2, sum3]
        sum_vec = vaddq_s32(sum_vec, vcombine_s32(lo, hi));
      }
      // Tail: fewer than 8 depth rows remain — zero-pad to 8.
      if (r < src_rows) {
        std::int16_t buf0[8]={}, buf1[8]={}, buf2[8]={}, buf3[8]={};
        int rem = src_rows - r;
        for (int k = 0; k < rem; k++) {
          buf0[k] = src0[k];
          buf1[k] = src1[k];
          buf2[k] = src2[k];
          buf3[k] = src3[k];
        }
        int16x8_t v0 = vld1q_s16(buf0), v1 = vld1q_s16(buf1);
        int16x8_t v2 = vld1q_s16(buf2), v3 = vld1q_s16(buf3);
        vst1q_s16(packed_ptr,      v0);
        vst1q_s16(packed_ptr + 8,  v1);
        vst1q_s16(packed_ptr + 16, v2);
        vst1q_s16(packed_ptr + 24, v3);
        packed_ptr += 32;
        int32x4_t s0 = vpaddlq_s16(v0); int32x4_t s1 = vpaddlq_s16(v1);
        int32x4_t s2 = vpaddlq_s16(v2); int32x4_t s3 = vpaddlq_s16(v3);
        int32x2_t r0 = vadd_s32(vget_low_s32(s0), vget_high_s32(s0));
        int32x2_t r1 = vadd_s32(vget_low_s32(s1), vget_high_s32(s1));
        int32x2_t r2 = vadd_s32(vget_low_s32(s2), vget_high_s32(s2));
        int32x2_t r3 = vadd_s32(vget_low_s32(s3), vget_high_s32(s3));
        int32x2_t lo = vpadd_s32(r0, r1);
        int32x2_t hi = vpadd_s32(r2, r3);
        sum_vec = vaddq_s32(sum_vec, vcombine_s32(lo, hi));
      }

      if (sums_ptr) {
        // Extract 4 per-column sums and store.
        int32_t tmp[4];
        vst1q_s32(tmp, sum_vec);
        for (int i = 0; i < 4 && (block_col + i) < src_matrix.layout.cols; i++)
          sums_ptr[i] = tmp[i];
      }
    }
  }
};

// Pack int8 (or uint8) source with kColMajor/8/4 layout for mixed-precision.
// Used as LHS for i8×i16→i16 kernels. 4 rows × 8 int8 depth = 32 bytes/step.
template <typename Scalar>
struct PackImpl<Path::kNeon,
                FixedKernelLayout<Order::kColMajor, 8, 4>,
                Scalar, std::int8_t, std::int32_t,
                Order::kColMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value, "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    profiler::ScopeLabel label("Pack (int8 kNeon ColMajor 8x4)");
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);

    const int src_stride = src_matrix.layout.stride;
    const int src_rows = src_matrix.layout.rows;
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[8];
    memset(zerobuf, static_cast<std::uint8_t>(src_matrix.zero_point),
           sizeof(zerobuf));

    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      const Scalar* src0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src1 = src0 + src_stride;
      const Scalar* src2 = src1 + src_stride;
      const Scalar* src3 = src2 + src_stride;
      int inc0 = 8, inc1 = 8, inc2 = 8, inc3 = 8;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols)     { src0 = zerobuf; inc0 = 0; }
        if (block_col >= src_matrix.layout.cols - 1) { src1 = zerobuf; inc1 = 0; }
        if (block_col >= src_matrix.layout.cols - 2) { src2 = zerobuf; inc2 = 0; }
        if (block_col >= src_matrix.layout.cols - 3) { src3 = zerobuf; inc3 = 0; }
      }

      std::int8_t* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;

      int32x4_t sum_vec = vdupq_n_s32(0);

      int r = 0;
      for (; r + 8 <= src_rows; r += 8) {
        int8x8_t v0 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(src0)),
            vdup_n_u8(static_cast<std::uint8_t>(kInputXor))));
        int8x8_t v1 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(src1)),
            vdup_n_u8(static_cast<std::uint8_t>(kInputXor))));
        int8x8_t v2 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(src2)),
            vdup_n_u8(static_cast<std::uint8_t>(kInputXor))));
        int8x8_t v3 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(src3)),
            vdup_n_u8(static_cast<std::uint8_t>(kInputXor))));
        src0 += inc0; src1 += inc1; src2 += inc2; src3 += inc3;
        vst1_s8(packed_ptr,      v0);
        vst1_s8(packed_ptr + 8,  v1);
        vst1_s8(packed_ptr + 16, v2);
        vst1_s8(packed_ptr + 24, v3);
        packed_ptr += 32;
        if (sums_ptr) {
          // Column sums: reduce each int8x8 column to one int32.
          // vpaddl(vpaddl(v)) gives 2-element int32; then vadd scalar.
          int16x4_t ps0 = vpaddl_s8(v0), ps1 = vpaddl_s8(v1);
          int16x4_t ps2 = vpaddl_s8(v2), ps3 = vpaddl_s8(v3);
          int32x2_t qs0 = vpaddl_s16(ps0), qs1 = vpaddl_s16(ps1);
          int32x2_t qs2 = vpaddl_s16(ps2), qs3 = vpaddl_s16(ps3);
          // Combine: col0 in lane0 of lo, col1 in lane1, etc.
          int32x2_t lo = vpadd_s32(qs0, qs1);
          int32x2_t hi = vpadd_s32(qs2, qs3);
          sum_vec = vaddq_s32(sum_vec, vcombine_s32(lo, hi));
        }
      }
      if (r < src_rows) {
        Scalar buf0[8]={}, buf1[8]={}, buf2[8]={}, buf3[8]={};
        int rem = src_rows - r;
        for (int k = 0; k < rem; k++) {
          buf0[k] = src0[k]; buf1[k] = src1[k];
          buf2[k] = src2[k]; buf3[k] = src3[k];
        }
        auto xorv = vdup_n_u8(static_cast<std::uint8_t>(kInputXor));
        int8x8_t v0 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(buf0)), xorv));
        int8x8_t v1 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(buf1)), xorv));
        int8x8_t v2 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(buf2)), xorv));
        int8x8_t v3 = vreinterpret_s8_u8(veor_u8(
            vld1_u8(reinterpret_cast<const std::uint8_t*>(buf3)), xorv));
        vst1_s8(packed_ptr,      v0);
        vst1_s8(packed_ptr + 8,  v1);
        vst1_s8(packed_ptr + 16, v2);
        vst1_s8(packed_ptr + 24, v3);
        if (sums_ptr) {
          int16x4_t ps0 = vpaddl_s8(v0), ps1 = vpaddl_s8(v1);
          int16x4_t ps2 = vpaddl_s8(v2), ps3 = vpaddl_s8(v3);
          int32x2_t qs0 = vpaddl_s16(ps0), qs1 = vpaddl_s16(ps1);
          int32x2_t qs2 = vpaddl_s16(ps2), qs3 = vpaddl_s16(ps3);
          int32x2_t lo = vpadd_s32(qs0, qs1);
          int32x2_t hi = vpadd_s32(qs2, qs3);
          sum_vec = vaddq_s32(sum_vec, vcombine_s32(lo, hi));
        }
      }

      if (sums_ptr) {
        int32_t tmp[4];
        vst1q_s32(tmp, sum_vec);
        for (int i = 0; i < 4 && (block_col + i) < src_matrix.layout.cols; i++)
          sums_ptr[i] = tmp[i];
      }
    }
  }
};

#endif  // defined(_MSC_VER) && defined(_M_ARM64)

#if RUY_PLATFORM_NEON_32 && RUY_OPT(ASM) && !defined(_MSC_VER)
// The 32-bit float kernel is 4 rows X 2 columns, so we need an additional
// partial specialization for the RHS, which has a FixedKernelLayout with 2
// columns.
template <typename Scalar>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kColMajor, 16, 2>, Scalar,
                std::int8_t, std::int32_t, Order::kColMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;
  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 2, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 2) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      int src_inc0 = 16;
      int src_inc1 = 16;
      if (block_col >= src_matrix.layout.cols - 2) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      PackParams8bit params;
      MakePackParams8bit(src_ptr0, src_ptr1, nullptr, nullptr, sums_ptr,
                         packed_ptr, src_inc0, src_inc1, -1, -1,
                         src_matrix.layout.rows, src_matrix.zero_point,
                         kInputXor, &params);
      Pack8bitColMajorForNeon2Cols(params);
    }
  }
};
#endif  // (RUY_PLATFORM_NEON_32) && RUY_OPT(ASM) && !defined(_MSC_VER)

#if RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && \
    (!defined(_MSC_VER) || defined(_M_ARM64))
template <typename Scalar>
struct PackImpl<Path::kNeonDotprod, FixedKernelLayout<Order::kColMajor, 4, 8>,
                Scalar, std::int8_t, std::int32_t, Order::kColMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & ~7) +
          ((block_col & 4) * 4);
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      if (__builtin_expect(tuning == Tuning::kA55ish, true)) {
        Pack8bitColMajorForNeonDotprodA55ish(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, sums_ptr, kInputXor);
      } else {
        Pack8bitColMajorForNeonDotprod(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, sums_ptr, kInputXor);
      }
    }
  }
};
#endif  // RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && (!defined(_MSC_VER) || defined(_M_ARM64))

#if RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && \
    (!defined(_MSC_VER) || defined(_M_ARM64))
void PackFloatColMajorForNeon(const float* src_ptr0, const float* src_ptr1,
                              const float* src_ptr2, const float* src_ptr3,
                              int src_inc0, int src_inc1, int src_inc2,
                              int src_inc3, int src_rows, float* packed_ptr);
void PackFloatColMajorForNeonA55ish(const float* src_ptr0,
                                    const float* src_ptr1,
                                    const float* src_ptr2,
                                    const float* src_ptr3, int src_inc0,
                                    int src_inc1, int src_inc2, int src_inc3,
                                    int src_rows, float* packed_ptr);

#elif RUY_PLATFORM_NEON_32 && RUY_OPT(ASM) && !defined(_MSC_VER)
void PackFloatColMajorForNeon(const float* src_ptr0, const float* src_ptr1,
                              const float* src_ptr2, const float* src_ptr3,
                              int src_inc, int src_rows, float* packed_ptr,
                              int stride);
#endif  // RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && (!defined(_MSC_VER) || defined(_M_ARM64))

#if (RUY_PLATFORM_NEON_32 || RUY_PLATFORM_NEON_64) && RUY_OPT(ASM) && \
    (!defined(_MSC_VER) || defined(_M_ARM64))

template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
                float, float, Order::kColMajor> {
  static void Run(Tuning tuning, const Mat<float>& src_matrix,
                  PMat<float>* packed_matrix, int start_col, int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr = packed_matrix->data +
                          packed_matrix->layout.stride * (block_col & ~7) +
                          ((block_col & 4));
#if RUY_PLATFORM_NEON_64
      if (__builtin_expect(tuning == Tuning::kA55ish, true)) {
        PackFloatColMajorForNeonA55ish(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                       src_inc0, src_inc1, src_inc2, src_inc3,
                                       src_matrix.layout.rows, packed_ptr);
      } else {
        PackFloatColMajorForNeon(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                 src_inc0, src_inc1, src_inc2, src_inc3,
                                 src_matrix.layout.rows, packed_ptr);
      }
#else
      (void)tuning;
      // Encode each of src_inc0, ..., src_inc3 in lowest 4 bits of src_inc
      // to save on registers (we have fewer general purpose registers in
      // 32-bit ARM than in 64-bit ARM). For the 64-bit case, we pass four
      // values that are each either 16 or 0 and use them directly. For the
      // 32-bit case, bits 0, 1, 2, and 3 are used to determine if we should
      // use the value 16 (bit is set) or 0 (bit is not set) for the
      // respective increment value.
      std::int64_t src_inc = 0;
      src_inc += src_inc0 == 16 ? 1 : 0;
      src_inc += src_inc1 == 16 ? 2 : 0;
      src_inc += src_inc2 == 16 ? 4 : 0;
      src_inc += src_inc3 == 16 ? 8 : 0;
      const int kOutputStride = 32;
      PackFloatColMajorForNeon(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc,
                               src_matrix.layout.rows, packed_ptr,
                               kOutputStride);
#endif  // RUY_PLATFORM_NEON_64
    }
  }
};

#if RUY_PLATFORM_NEON_32
// The 32-bit float kernel is 8 rows X 4 columns, so we need an additional
// specialization for a FixedKernelLayout with 4 columns.
template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 1, 4>, float,
                float, float, Order::kColMajor> {
  static void Run(Tuning, const Mat<float>& src_matrix,
                  PMat<float>* packed_matrix, int start_col, int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * (block_col);
      // Encode each of src_inc0, ..., src_inc1 in lowest 4 bits of scrc_inc
      // to save registers.
      std::int64_t src_inc = 0;
      src_inc += src_inc0 == 16 ? 1 : 0;
      src_inc += src_inc1 == 16 ? 2 : 0;
      src_inc += src_inc2 == 16 ? 4 : 0;
      src_inc += src_inc3 == 16 ? 8 : 0;
      const int kOutputStride = 16;
      PackFloatColMajorForNeon(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc,
                               src_matrix.layout.rows, packed_ptr,
                               kOutputStride);
    }
  }
};
#endif  // (RUY_PLATFORM_NEON_32)
#endif  // (RUY_PLATFORM_NEON_64 || RUY_PLATFORM_NEON_32) && \
        // RUY_OPT(ASM)

#if RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && \
    (!defined(_MSC_VER) || defined(_M_ARM64))

template <typename Scalar>
struct PackImpl<Path::kNeonDotprod, FixedKernelLayout<Order::kColMajor, 4, 8>,
                Scalar, std::int8_t, std::int32_t, Order::kRowMajor> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsRowMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    std::int32_t* sums = packed_matrix->sums;
    std::memset(sums + start_col, 0, sizeof(sums[0]) * (end_col - start_col));
    Scalar zerobuf[8];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    int src_stride = src_matrix.layout.stride;
    // As the source matrix is row-major and the destination packed matrix is
    // column-major, there is no traversal order that will be optimal for both
    // so we choose to favor the source matrix with a row-major traversal order.
    // Loop over groups of 4 rows.
    for (int block_row = 0; block_row < packed_matrix->layout.rows;
         block_row += 4) {
      // src_ptr[0-3] shall point to the positions in the 4 rows of the source
      // matrix that we are loading from, and will be incremented by
      // src_inc[0-3] after each 4x8 block is loaded.
      // First we compute these src_ptr and src_inc values for the case where
      // there are 4 rows left to be loaded from in the source matrix ...
      const Scalar* src_ptr0 =
          src_matrix.data.get() + src_stride * block_row + start_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 8;
      std::int64_t src_inc1 = 8;
      std::int64_t src_inc2 = 8;
      std::int64_t src_inc3 = 8;
      // ... and now we adjust these values in case there are fewer than 4 rows
      // left to load from in the source matrix. In that case, we set the
      // corresponding src_ptr pointer to load from `zerobuf` and set src_inc
      // to 0 to avoid overrunning that small buffer.
      if (block_row >= src_matrix.layout.rows - 3) {
        if (block_row >= src_matrix.layout.rows - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_row >= src_matrix.layout.rows - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_row >= src_matrix.layout.rows - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_row >= src_matrix.layout.rows - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      // Let src_cols be the number of source matrix columns to handle.
      int src_cols = std::min(end_col, src_matrix.layout.cols) - start_col;
      std::int8_t* packed_ptr = packed_matrix->data +
                                packed_matrix->layout.stride * start_col +
                                8 * block_row;
      std::int32_t* sums_ptr = sums + start_col;
      Pack8bitRowMajorForNeonDotprod(
          src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1, src_inc2,
          src_inc3, src_cols, src_matrix.zero_point, packed_ptr,
          packed_matrix->layout.stride, sums_ptr, kInputXor);
    }
  }
};

#endif  // RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && (!defined(_MSC_VER) || defined(_M_ARM64))

#if RUY_PLATFORM_NEON

template <typename Scalar, int KernelCols>
struct PackImpl<Path::kNeon,
                FixedKernelLayout<Order::kColMajor, 16, KernelCols>, Scalar,
                std::int8_t, std::int32_t, Order::kRowMajor> {
  static void Run(Tuning, const Mat<Scalar>& src_matrix,
                  PMat<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    profiler::ScopeLabel label("Pack (KNeon, from row-major source)");
    static constexpr int kInputXor =
        std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;
    RUY_DCHECK_EQ(src_matrix.layout.order, Order::kRowMajor);
    RUY_DCHECK_EQ((end_col - start_col) % KernelCols, 0);
    std::int32_t* sums = packed_matrix->sums;
    std::memset(sums + start_col, 0, sizeof(sums[0]) * (end_col - start_col));
    int block_row = 0;
    for (; block_row < packed_matrix->layout.rows; block_row += 16) {
      int src_stride = src_matrix.layout.stride;
      int packed_stride = packed_matrix->layout.stride;
      const Scalar* src_ptr =
          src_matrix.data.get() + block_row * src_stride + start_col;
      std::int8_t* packed_ptr = packed_matrix->data +
                                start_col * packed_stride +
                                block_row * KernelCols;

      Pack8bitRowMajorForNeon(
          reinterpret_cast<const std::uint8_t*>(src_ptr), src_stride,
          src_matrix.layout.rows, src_matrix.layout.cols, block_row, start_col,
          end_col, packed_ptr, packed_stride, packed_matrix->zero_point, sums,
          kInputXor, KernelCols);
    }
  }
};
#endif

}  // namespace ruy

#endif  // RUY_RUY_PACK_ARM_H_
