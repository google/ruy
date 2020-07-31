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

// AVX doesn't have fused multiply-add so we define an inline function to be
// used in the common code following.
template <>
inline __m256 MulAdd<Path::kAvx>(const __m256& a, const __m256& b,
                                 const __m256& c) {
  const __m256 prod = _mm256_mul_ps(a, b);
  return _mm256_add_ps(prod, c);
}

}  // namespace intrin_utils
}  // namespace

void KernelFloatAvx(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel kAvx float");
  KernelFloatAvxCommon<Path::kAvx>(params);
}

void KernelFloatAvxSingleCol(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel kAvx float GEMV");
  KernelFloatAvxCommonSingleCol<Path::kAvx>(params);
}

#endif  //  RUY_PLATFORM_AVX && RUY_OPT(ASM)

}  // namespace ruy
