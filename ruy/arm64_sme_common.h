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


#ifndef RUY_ARM64_SME_COMMON_H
#define RUY_ARM64_SME_COMMON_H

#include "ruy/asm_helpers.h"
#include "ruy/check_macros.h"
#include "ruy/opt_set.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"

#if RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)

#include <arm_sve.h>

#ifndef FORCE_INLINE
#define FORCE_INLINE static inline __attribute__((always_inline))
#endif // FORCE_INLINE

namespace ruy
{
#define ARM64_SME_MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define ARM64_SME_MIN_UPPER_NEXT(a, b, next) (a) < ((b) + (next)) ? (a) : (b);

// The smstart/smstop calls are used to toggle between the Neon and SME/SVE.
// Toggling on SME/SVE overrides the d registers.
// Next 2 macros wraps the smstart/smstop calls - so we store all d registers to 
// a stack once SME/SVE code starts and load them back once SME/SVE code ends.

#define SMSTART(c_ptr)                                   \
  __asm__ __volatile__("stp d8,d9, [%[ptr]] \n\t"        \
                       "stp d10,d11, [%[ptr], #16] \n\t" \
                       "stp d12,d13, [%[ptr], #32] \n\t" \
                       "stp d14,d15, [%[ptr], #48] \n\t" \
                       "smstart \n\t"                    \
                       : : [ptr] "r"(c_ptr) : "memory");

#define SMSTOP(c_ptr)                                    \
  __asm__ __volatile__("smstop \n\t"                     \
                       "ldp d8,d9, [%[ptr]] \n\t"        \
                       "ldp d10,d11, [%[ptr], #16] \n\t" \
                       "ldp d12,d13, [%[ptr], #32] \n\t" \
                       "ldp d14,d15, [%[ptr], #48] \n\t" \
                       : : [ptr] "r"(c_ptr) : "memory");

  typedef long long sgemm_idx_t;
  typedef long long transpose_idx_t;
  typedef long long gemmi8_idx_t;
  typedef long long gemvi8_packed_idx_t;

  // Forward declaration of the floating point and fixed point params.
  template <int LhsCols, int RhsCols> struct KernelParamsFloat;
  template <int LhsCols, int RhsCols> struct KernelParams8bit;

  typedef KernelParamsFloat<16, 16> SME_RUY_Kernel_ParamsF32;
  typedef KernelParams8bit<16, 16> SME_RUY_Kernel_Params8Bits;

#endif // RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)

} // namespace ruy

#endif // RUY_ARM64_SME_COMMON_H
