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

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "ruy/check_macros.h"
#include "ruy/kernel_common.h"
#include "ruy/kernel_x86.h"
#include "ruy/opt_set.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"

#if RUY_PLATFORM_AVX && RUY_OPT(ASM)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM_AVX && RUY_OPT(ASM))

void KernelFloatAvx(const KernelParamsFloat<8, 8>&) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvxSingleCol(const KernelParamsFloat<8, 8>&) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM_AVX && RUY_OPT(ASM)

namespace {
namespace intrin_utils {

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

inline __m256 mm256_n_loadu_ps(int i, const float* src) {
  switch (i) {
    case 0:
      return _mm256_setzero_ps();
    case 1:
      return _mm256_setr_m128(_mm_setr_ps(src[0], .0f, .0f, .0f),
                              _mm_setzero_ps());
    case 2:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], .0f, .0f),
                              _mm_setzero_ps());
    case 3:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], src[2], .0f),
                              _mm_setzero_ps());
    case 4:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], src[2], src[3]),
                              _mm_setzero_ps());
    case 5:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], .0f, .0f,
                            .0f);
    case 6:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], src[5], .0f,
                            .0f);
    case 7:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], src[5],
                            src[6], .0f);
    case 8:
      return _mm256_loadu_ps(src);
    default:
      RUY_DCHECK_LT(i, 9);
      return _mm256_setzero_ps();
  }
}

inline void mm256_n_storeu_ps(float* dst, int residual_rows, const __m256 v) {
  for (int i = 0; i < residual_rows; ++i) {
    dst[i] = intrin_utils::mm256_get1_ps(v, i);
  }
}

// AVX doesn't have fused multiply-add so we define an inline function to be
// used in the common code following.
inline __m256 mm256_fmadd_ps(const __m256& a, const __m256& b,
                             const __m256& c) {
  const __m256 prod = _mm256_mul_ps(a, b);
  return _mm256_add_ps(prod, c);
}

}  // namespace intrin_utils
}  // namespace

#include "kernel_x86_common.inc"

void KernelFloatAvx(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel kAvx float");
  KernelFloatx86Common(params);
}

void KernelFloatAvxSingleCol(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel kAvx float GEMV");
  KernelFloatx86CommonSingleCol(params);
}

#endif  //  RUY_PLATFORM_AVX && RUY_OPT(ASM)

}  // namespace ruy
