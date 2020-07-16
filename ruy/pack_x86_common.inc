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


#if ((RUY_PLATFORM_AVX || RUY_PLATFORM_AVX2_FMA) && RUY_OPT(ASM))

inline void PackFloatColMajorForx86CommonPacker(const float* src_ptr,
                                           const float* zerobuf, int src_stride,
                                           int remaining_src_cols, int src_rows,
                                           float* packed_ptr,
                                           float* trailing_buf) {
  RUY_DCHECK_EQ(PackImplFloatx86::Layout::kCols, 8);
  RUY_DCHECK_EQ(PackImplFloatx86::Layout::kRows, 1);

  // This packing amounts to transposition of 8x8 blocks.
  static constexpr int kPackCols = 8;  // Source cols packed together.
  static constexpr int kPackRows = 8;  // Short input is padded.

  const float* src_ptr0 = src_ptr;
  const float* src_ptr1 = src_ptr0 + src_stride;
  const float* src_ptr2 = src_ptr1 + src_stride;
  const float* src_ptr3 = src_ptr2 + src_stride;
  const float* src_ptr4 = src_ptr3 + src_stride;
  const float* src_ptr5 = src_ptr4 + src_stride;
  const float* src_ptr6 = src_ptr5 + src_stride;
  const float* src_ptr7 = src_ptr6 + src_stride;
  std::int64_t src_inc0 = 8;
  std::int64_t src_inc1 = 8;
  std::int64_t src_inc2 = 8;
  std::int64_t src_inc3 = 8;
  std::int64_t src_inc4 = 8;
  std::int64_t src_inc5 = 8;
  std::int64_t src_inc6 = 8;
  std::int64_t src_inc7 = 8;
  // Handle cases where source does not have kPackDim (8) columns.
  if (remaining_src_cols < kPackCols) {
    if (remaining_src_cols <= 0) {
      src_ptr0 = zerobuf;
      src_inc0 = 0;
    }
    if (remaining_src_cols <= 1) {
      src_ptr1 = zerobuf;
      src_inc1 = 0;
    }
    if (remaining_src_cols <= 2) {
      src_ptr2 = zerobuf;
      src_inc2 = 0;
    }
    if (remaining_src_cols <= 3) {
      src_ptr3 = zerobuf;
      src_inc3 = 0;
    }
    if (remaining_src_cols <= 4) {
      src_ptr4 = zerobuf;
      src_inc4 = 0;
    }
    if (remaining_src_cols <= 5) {
      src_ptr5 = zerobuf;
      src_inc5 = 0;
    }
    if (remaining_src_cols <= 6) {
      src_ptr6 = zerobuf;
      src_inc6 = 0;
    }
    src_ptr7 = zerobuf;
    src_inc7 = 0;
  }

  for (int k = 0; k < src_rows; k += kPackRows) {
    const int available_src_rows = src_rows - k;
    // Effectively,
    // available_src_rows = std::max(0, std::min(kPackDim, src_rows - k));
    // but treat each case separately.
    if (available_src_rows >= kPackRows) {
      __m256 t0, t1, t2, t3, t4, t5, t6, t7;
      __m256 r0, r1, r2, r3, r4, r5, r6, r7;

      t0 = _mm256_loadu_ps(src_ptr0);
      t4 = _mm256_loadu_ps(src_ptr4);
      t1 = _mm256_loadu_ps(src_ptr1);
      t5 = _mm256_loadu_ps(src_ptr5);
      t2 = _mm256_loadu_ps(src_ptr2);
      t6 = _mm256_loadu_ps(src_ptr6);
      t3 = _mm256_loadu_ps(src_ptr3);
      t7 = _mm256_loadu_ps(src_ptr7);

      r0 = _mm256_unpacklo_ps(t0, t1);
      r4 = _mm256_unpacklo_ps(t4, t5);
      r2 = _mm256_unpackhi_ps(t0, t1);
      r6 = _mm256_unpackhi_ps(t4, t5);
      r1 = _mm256_unpacklo_ps(t2, t3);
      r5 = _mm256_unpacklo_ps(t6, t7);
      r3 = _mm256_unpackhi_ps(t2, t3);
      r7 = _mm256_unpackhi_ps(t6, t7);

      t0 = Mm256UnpackloPsx2(r0, r1);
      t4 = Mm256UnpackloPsx2(r4, r5);
      t2 = Mm256UnpackhiPsx2(r0, r1);
      t6 = Mm256UnpackhiPsx2(r4, r5);
      t1 = Mm256UnpackloPsx2(r2, r3);
      t5 = Mm256UnpackloPsx2(r6, r7);
      t3 = Mm256UnpackhiPsx2(r2, r3);
      t7 = Mm256UnpackhiPsx2(r6, r7);

      // The preceding sets of rearrangement operations interleaved by 4 bytes
      // and then by 8 bytes *within* lanes. The following set interleave by 16
      // bytes (128-bit), operating *between* AVX lanes. For instance (t0, t4)
      // are interleaved to create (r0, r1). This complexity follows from the
      // way that AVX is centered around MM 128-bit lanes.
      r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
      r4 = _mm256_permute2f128_ps(t1, t5, 0x20);
      r1 = _mm256_permute2f128_ps(t0, t4, 0x31);
      r5 = _mm256_permute2f128_ps(t1, t5, 0x31);
      r2 = _mm256_permute2f128_ps(t2, t6, 0x20);
      r6 = _mm256_permute2f128_ps(t3, t7, 0x20);
      r3 = _mm256_permute2f128_ps(t2, t6, 0x31);
      r7 = _mm256_permute2f128_ps(t3, t7, 0x31);

      _mm256_storeu_ps(packed_ptr + 0 * 8, r0);
      _mm256_storeu_ps(packed_ptr + 2 * 8, r4);
      _mm256_storeu_ps(packed_ptr + 4 * 8, r1);
      _mm256_storeu_ps(packed_ptr + 6 * 8, r5);
      _mm256_storeu_ps(packed_ptr + 1 * 8, r2);
      _mm256_storeu_ps(packed_ptr + 3 * 8, r6);
      _mm256_storeu_ps(packed_ptr + 5 * 8, r3);
      _mm256_storeu_ps(packed_ptr + 7 * 8, r7);
    } else if (available_src_rows > 0) {
      const __m256i series = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
      const __m256i row_mask_v = Mm256CompareGreaterThan(_mm256_set1_epi32(available_src_rows), series);

      __m256 t0, t1, t2, t3, t4, t5, t6, t7;
      __m256 r0, r1, r2, r3, r4, r5, r6, r7;

      t0 = _mm256_maskload_ps(src_ptr0, row_mask_v);
      t4 = _mm256_maskload_ps(src_ptr4, row_mask_v);
      t1 = _mm256_maskload_ps(src_ptr1, row_mask_v);
      t5 = _mm256_maskload_ps(src_ptr5, row_mask_v);
      t2 = _mm256_maskload_ps(src_ptr2, row_mask_v);
      t6 = _mm256_maskload_ps(src_ptr6, row_mask_v);
      t3 = _mm256_maskload_ps(src_ptr3, row_mask_v);
      t7 = _mm256_maskload_ps(src_ptr7, row_mask_v);

      r0 = _mm256_unpacklo_ps(t0, t1);
      r4 = _mm256_unpacklo_ps(t4, t5);
      r2 = _mm256_unpackhi_ps(t0, t1);
      r6 = _mm256_unpackhi_ps(t4, t5);
      r1 = _mm256_unpacklo_ps(t2, t3);
      r5 = _mm256_unpacklo_ps(t6, t7);
      r3 = _mm256_unpackhi_ps(t2, t3);
      r7 = _mm256_unpackhi_ps(t6, t7);

      t0 = Mm256UnpackloPsx2(r0, r1);
      t4 = Mm256UnpackloPsx2(r4, r5);
      t2 = Mm256UnpackhiPsx2(r0, r1);
      t6 = Mm256UnpackhiPsx2(r4, r5);
      t1 = Mm256UnpackloPsx2(r2, r3);
      t5 = Mm256UnpackloPsx2(r6, r7);
      t3 = Mm256UnpackhiPsx2(r2, r3);
      t7 = Mm256UnpackhiPsx2(r6, r7);

      // The preceding sets of rearrangement operations interleaved by 4 bytes
      // and then by 8 bytes *within* lanes. The following set interleave by 16
      // bytes (128-bit), operating *between* AVX lanes. For instance (t0, t4)
      // are interleaved to create (r0, r1). This complexity follows from the
      // way that AVX is centered around MM 128-bit lanes.
      r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
      r4 = _mm256_permute2f128_ps(t1, t5, 0x20);
      r1 = _mm256_permute2f128_ps(t0, t4, 0x31);
      r5 = _mm256_permute2f128_ps(t1, t5, 0x31);
      r2 = _mm256_permute2f128_ps(t2, t6, 0x20);
      r6 = _mm256_permute2f128_ps(t3, t7, 0x20);
      r3 = _mm256_permute2f128_ps(t2, t6, 0x31);
      // r7 no longer needed.

      _mm256_storeu_ps(trailing_buf + 0 * 8, r0);
      _mm256_storeu_ps(trailing_buf + 2 * 8, r4);
      _mm256_storeu_ps(trailing_buf + 4 * 8, r1);
      _mm256_storeu_ps(trailing_buf + 6 * 8, r5);
      _mm256_storeu_ps(trailing_buf + 1 * 8, r2);
      _mm256_storeu_ps(trailing_buf + 3 * 8, r6);
      _mm256_storeu_ps(trailing_buf + 5 * 8, r3);
      // No store to (trailing_buf + 7 * 8), space not allocated.
    }

    packed_ptr += kPackRows * kPackCols;
    src_ptr0 += src_inc0;
    src_ptr1 += src_inc1;
    src_ptr2 += src_inc2;
    src_ptr3 += src_inc3;
    src_ptr4 += src_inc4;
    src_ptr5 += src_inc5;
    src_ptr6 += src_inc6;
    src_ptr7 += src_inc7;
  }
}
#endif  //  (RUY_PLATFORM_AVX || RUY_PLATFORM_AVX2_FMA) && RUY_OPT(ASM)
