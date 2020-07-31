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

#include <cstdint>
#include <cstring>

#include "ruy/check_macros.h"
#include "ruy/opt_set.h"
#include "ruy/pack_x86.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"

#if RUY_PLATFORM_AVX && RUY_OPT(INTRINSICS)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM_AVX && RUY_OPT(ASM))

void PackFloatColMajorForAvx(const float*, const float*, int, int, int,
                             float*) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM_AVX && RUY_OPT(ASM)

using PackImplFloatAvx =
    PackImpl<Path::kAvx, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
             float, float, Order::kColMajor>;

namespace {

// Use a generic AVX intrinsic for greater-than comparison.
template <>
inline __m256i CompareGreaterThan<Path::kAvx>(const __m256i& a,
                                              const __m256i& b) {
  constexpr int kGreaterThanSignalling = 14;
  const __m256 v = _mm256_cmp_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b),
                                 kGreaterThanSignalling);
  return _mm256_cvtps_epi32(v);
}

}  // namespace.

void PackFloatColMajorForAvx(const float* src_ptr, const float* zerobuf,
                             int src_stride, int remaining_src_cols,
                             int src_rows, float* packed_ptr) {
  profiler::ScopeLabel label("Pack kAvx float");
  static constexpr int kPackCols = 8;  // Source cols packed together.
  static constexpr int kPackRows = 8;  // Short input is padded.
  float trailing_buf[(kPackRows - 1) * kPackCols];
  if (remaining_src_cols < 8) {
    memset(trailing_buf, 0, sizeof(trailing_buf));
  }
  PackFloatColMajorForAvxCommonPacker<PackImplFloatAvx, Path::kAvx>(
      src_ptr, zerobuf, src_stride, remaining_src_cols, src_rows, packed_ptr,
      trailing_buf);

  const int trailing_rows = src_rows & (kPackRows - 1);
  if (trailing_rows > 0) {
    const int non_trailing_rows = src_rows & ~(kPackRows - 1);
    memcpy(packed_ptr + kPackCols * non_trailing_rows, trailing_buf,
           kPackCols * trailing_rows * sizeof(float));
  }
}

#endif  // RUY_PLATFORM_AVX && RUY_OPT(INTRINSICS)

}  // namespace ruy
