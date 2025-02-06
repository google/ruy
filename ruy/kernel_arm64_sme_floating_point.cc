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

#include <cstdint>

#include "ruy/asm_helpers.h"
#include "ruy/check_macros.h"
#include "ruy/kernel_arm.h"
#include "ruy/opt_set.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"

// When using SME for floating point we logically "split" the processing grid into 4 tiles.
// Performance wise - having wide output matrix the grid performs best processing 4 sequential tiles.
// We use a "greedy" approach - in case the remaining destination elements fill the greed sequentially
// we process them using sme_sgemm_NT_4x1_batch. Next we try sme_sgemm_NT_2x2_batch and
// sme_sgemm_NT_2x1_batch (depending on the destination height), and last sme_sgemm_NT_1xN_batch.

#if RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)
#include "ruy/arm64_sme_common.h"
namespace ruy
{
  // In case we process only 1 or 2 tiles (sme_sgemm_NT_2x1_batch or sme_sgemm_NT_1xN_batch), 
  // and the depth is bigger than SPLIT_THRESHOLD we use partial sums over 4 tiles and 
  // add them at the end.
  static const sgemm_idx_t SPLIT_THRESHOLD = 48;
  static const uint32_t one_f = 0x3f800000;
  struct SME_Kernel_Params
  {
    // Original RUY params are used as read only.
    const SME_RUY_Kernel_ParamsF32* ruy_params; 
    
    // Next members are modified per iteration and inside the batches.
    sgemm_idx_t num_rows;
    sgemm_idx_t num_cols;
    const float *lhs_ptr;
    const float *rhs_ptr;
          float *dst_ptr;
    std::int32_t start_row;
    std::int32_t start_col;
  };

  __attribute__((noinline))
  static void sme_sgemm_NT_4x1_batch(const SME_Kernel_Params &params)
  {
    const float *lhs_ptr = params.lhs_ptr;
    const float *rhs_ptr = params.rhs_ptr;
          float *dst_ptr = params.dst_ptr;
    sgemm_idx_t lhs_stride_bytes = params.ruy_params->lhs_stride;
    sgemm_idx_t rhs_stride_bytes = params.ruy_params->rhs_stride;
    sgemm_idx_t dst_stride_bytes = params.ruy_params->dst_stride  << 2;

    __asm__ __volatile__("zero    {za} \n\t"
                         "whilelt pn8.s, xzr, %[num_rows], vlx4 \n\t"
                         "pext    { p0.s, p1.s }, pn8[0] \n\t"
                         "pext    { p2.s, p3.s }, pn8[1] \n\t"
                         "whilelt p4.s, xzr, %[num_cols] \n\t"
                         :
                         : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_cols)
                         : "memory", "cc");

    for (sgemm_idx_t depth = 0; depth < params.ruy_params->depth; ++depth)
    {
      // Calculating the GEMM
      __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[lhs_ptr]] \n\t"
                           "ld1w    { z4.s }, p4/z, [%[rhs_ptr]] \n\t"
                           "fmopa   za0.s, p4/m, p0/m, z4.s, z0.s \n\t"
                           "fmopa   za1.s, p4/m, p1/m, z4.s, z1.s \n\t"
                           "fmopa   za2.s, p4/m, p2/m, z4.s, z2.s \n\t"
                           "fmopa   za3.s, p4/m, p3/m, z4.s, z3.s \n\t"
                           "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                           "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                           : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                           : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                           : "memory");
    }

    // Adding the bias
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      const float *bias_ptr = NULL;
      if (!(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "dup      z4.s, %w[one_f] \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                             : "memory");
      }
      else
      {
        bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[bias_ptr]] \n\t"
                             "dup     z0.s, %w[one_f] \n\t"
                             "mov     z1.d, z0.d \n\t"
                             "mov     z2.d, z0.d \n\t"
                             "mov     z3.d, z0.d \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                             : "memory");
      }
      // As the GEMM results are still in the ZA (accumulator registers),
      // we add the bias using macc operation: ZA += 1 (z4) * bias (z0-z3).
      __asm__ __volatile__("fmopa   za0.s, p4/m, p0/m, z4.s, z0.s \n\t"
                           "fmopa   za1.s, p4/m, p1/m, z4.s, z1.s \n\t"
                           "fmopa   za2.s, p4/m, p2/m, z4.s, z2.s \n\t"
                           "fmopa   za3.s, p4/m, p3/m, z4.s, z3.s \n\t"
                           : : : "memory");
    }

    // Duplicating clamp values
    __asm__ __volatile__("dup z12.s, %w[clamp_min] \n\t"
                         "dup z13.s, %w[clamp_max] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max));
    
    // store ZA data into C
    sgemm_idx_t i = params.num_cols;
    register uint32_t za_index asm("w12");
    za_index = 0;
    while (i >= 2)
    {
      // Moving the accumulators (after bias) to z registers, clamping and storing.
      __asm__ __volatile__("mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
                           "mova    { z4.b, z5.b, z6.b, z7.b }, za0h.b[%w[za_index], 4:7] \n\t"
                           "fclamp  { z0.s-z3.s }, z12.s, z13.s \n\t"
                           "fclamp  { z4.s-z7.s }, z12.s, z13.s \n\t"
                           "st1w    { z0.s, z1.s, z2.s, z3.s }, pn8, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           "st1w    { z4.s, z5.s, z6.s, z7.s }, pn8, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           : [dst_ptr] "+r"(dst_ptr)
                           : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
      za_index += 8;
      i -= 2;
    }
    if (i)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
                           "fclamp  { z0.s-z3.s }, z12.s, z13.s \n\t"
                           "st1w    { z0.s, z1.s, z2.s, z3.s }, pn8, [%[dst_ptr]] \n\t"
                           :
                           : [za_index] "r"(za_index), [dst_ptr] "r"(dst_ptr), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
    }
  }


  __attribute__((noinline))
  static void sme_sgemm_NT_2x2_batch(const SME_Kernel_Params &params)
  {
    const float *lhs_ptr = params.lhs_ptr;
    const float *rhs_ptr = params.rhs_ptr;
          float *dst_ptr = params.dst_ptr;
    sgemm_idx_t lhs_stride_bytes = params.ruy_params->lhs_stride;
    sgemm_idx_t rhs_stride_bytes = params.ruy_params->rhs_stride;
    sgemm_idx_t dst_stride_bytes = params.ruy_params->dst_stride  << 2;

    __asm__ __volatile__("zero    {za} \n\t"
                         "whilelt pn8.s, xzr, %[num_rows], vlx2 \n\t"
                         "pext    { p0.s, p1.s }, pn8[0] \n\t"
                         "whilelt pn9.s, xzr, %[num_cols], vlx2 \n\t"
                         "pext    { p2.s, p3.s }, pn9[0] \n\t"
                         :
                         : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_cols)
                         : "memory", "cc");
    
    for (sgemm_idx_t depth = 0; depth < params.ruy_params->depth; ++depth)
    {
      __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[lhs_ptr]] \n\t"
                           "ld1w    { z2.s, z3.s }, pn9/z, [%[rhs_ptr]] \n\t"
                           "fmopa   za0.s, p2/m, p0/m, z2.s, z0.s \n\t"
                           "fmopa   za1.s, p2/m, p1/m, z2.s, z1.s \n\t"
                           "fmopa   za2.s, p3/m, p0/m, z3.s, z0.s \n\t"
                           "fmopa   za3.s, p3/m, p1/m, z3.s, z1.s \n\t"
                           "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                           "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                           : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                           : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                           : "memory");
    }
    
    sgemm_idx_t i = params.num_cols;
    uint64_t za_rows = svcntb();
    register uint32_t za_index asm("w12") = 0;

    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      const float *bias_ptr = NULL;
      if (!(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "dup     z2.s, %w[one_f] \n\t"
                             "mov     z3.d, z2.d \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                             : "memory");
      }
      else
      {
        bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z2.s, z3.s }, pn9/z, [%[bias_ptr]] \n\t"
                             "dup     z0.s, %w[one_f] \n\t"
                             "mov     z1.d, z0.d \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                             : "memory");
      }

      __asm__ __volatile__("fmopa   za0.s, p2/m, p0/m, z2.s, z0.s \n\t"
                           "fmopa   za1.s, p2/m, p1/m, z2.s, z1.s \n\t"
                           "fmopa   za2.s, p3/m, p0/m, z3.s, z0.s \n\t"
                           "fmopa   za3.s, p3/m, p1/m, z3.s, z1.s \n\t"
                           :
                           :
                           : "memory");
    }
    
     __asm__ __volatile__("dup z12.s, %w[clamp_min] \n\t"
                          "dup z13.s, %w[clamp_max] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max));
    
    while (i >= 2)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                           "mova    { z2.b, z3.b }, za0h.b[%w[za_index], 4:5] \n\t"
                           "fclamp  { z0.s-z3.s }, z12.s, z13.s \n\t"
                           "st1w    { z0.s, z1.s }, pn8, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           "st1w    { z2.s, z3.s }, pn8, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           : [dst_ptr] "+r"(dst_ptr)
                           : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
      za_index += 8;
      if (za_index >= za_rows)
      {
        za_index = 2;
      }
      i -= 2;
    }
    
    if (i)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                           "fclamp  { z0.s, z1.s }, z12.s, z13.s \n\t"
                           "st1w    { z0.s, z1.s }, pn8, [%[dst_ptr]] \n\t"
                           :
                           : [za_index] "r"(za_index), [dst_ptr] "r"(dst_ptr), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
    }
  }

  __attribute__((noinline))
  static void sme_sgemm_NT_2x1_batch(const SME_Kernel_Params &params)
  {
    const float *lhs_ptr = params.lhs_ptr;
    const float *rhs_ptr = params.rhs_ptr;
          float *dst_ptr = params.dst_ptr;
    sgemm_idx_t lhs_stride_bytes = params.ruy_params->lhs_stride;
    sgemm_idx_t rhs_stride_bytes = params.ruy_params->rhs_stride;
    sgemm_idx_t dst_stride_bytes = params.ruy_params->dst_stride  << 2;

    __asm__ __volatile__("zero    {za} \n\t"
                         "whilelt pn8.s, xzr, %[num_rows], vlx2 \n\t"
                         "pext    { p0.s, p1.s }, pn8[0] \n\t"
                         "whilelt p2.s, xzr, %[num_cols] \n\t"
                         :
                         : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_cols)
                         : "memory", "cc");

    if (params.ruy_params->depth >= SPLIT_THRESHOLD)
    {
      sgemm_idx_t depth = params.ruy_params->depth;
      while (depth >= 2)
      {
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[lhs_ptr]] \n\t"
                             "ld1w    { z2.s }, p2/z, [%[rhs_ptr]] \n\t"
                             "fmopa   za0.s, p2/m, p0/m, z2.s, z0.s \n\t"
                             "fmopa   za1.s, p2/m, p1/m, z2.s, z1.s \n\t"
                             "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                             "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                             "ld1w    { z0.s, z1.s }, pn8/z, [%[lhs_ptr]] \n\t"
                             "ld1w    { z2.s }, p2/z, [%[rhs_ptr]] \n\t"
                             "fmopa   za2.s, p2/m, p0/m, z2.s, z0.s \n\t"
                             "fmopa   za3.s, p2/m, p1/m, z2.s, z1.s \n\t"
                             "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                             "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                             : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                             : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                             : "memory");
        depth -= 2;
      }
      if (depth)
      {
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[lhs_ptr]] \n\t"
                             "ld1w    { z2.s }, p2/z, [%[rhs_ptr]] \n\t"
                             "fmopa   za0.s, p2/m, p0/m, z2.s, z0.s \n\t"
                             "fmopa   za1.s, p2/m, p1/m, z2.s, z1.s \n\t"
                             :
                             : [lhs_ptr] "r"(lhs_ptr), [rhs_ptr] "r"(rhs_ptr)
                             : "memory");
      }
      register uint32_t za_index asm("w11");
      uint32_t element_count = svcntw();
      for (za_index = 0; za_index < element_count; za_index += 4)
      {
        __asm__ __volatile__("mova    { z0.s, z1.s, z2.s, z3.s }, za.s[%w[za_index], 2] \n\t"
                             "mova    { z4.s, z5.s, z6.s, z7.s }, za.s[%w[za_index], 3] \n\t"
                             "fadd    za.s[%w[za_index], 0], { z0.s, z1.s, z2.s, z3.s } \n\t"
                             "fadd    za.s[%w[za_index], 1], { z4.s, z5.s, z6.s, z7.s } \n\t"
                             :
                             : [za_index] "r"(za_index)
                             : "memory");
      }
    }
    else
    {
      for (sgemm_idx_t depth = 0; depth < params.ruy_params->depth; ++depth)
      {
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[lhs_ptr]] \n\t"
                             "ld1w    { z2.s }, p2/z, [%[rhs_ptr]] \n\t"
                             "fmopa   za0.s, p2/m, p0/m, z2.s, z0.s \n\t"
                             "fmopa   za1.s, p2/m, p1/m, z2.s, z1.s \n\t"
                             "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                             "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                             : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                             : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                             : "memory");
      }
    }

    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      const float *bias_ptr = NULL;
      if (!(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "dup      z2.s, %w[one_f] \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                             : "memory");
      }
      else
      {
        bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z2.s }, p2/z, [%[bias_ptr]] \n\t"
                             "dup     z0.s, %w[one_f] \n\t"
                             "mov     z1.d, z0.d \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                             : "memory");
      }

      __asm__ __volatile__("fmopa   za0.s, p2/m, p0/m, z2.s, z0.s \n\t"
                           "fmopa   za1.s, p2/m, p1/m, z2.s, z1.s \n\t"
                           : : : "memory");
    }

     __asm__ __volatile__("dup z12.s, %w[clamp_min] \n\t"
                          "dup z13.s, %w[clamp_max] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max));
    
    sgemm_idx_t i = params.num_cols;
    register uint32_t za_index asm("w12");
    za_index = 0;
    while (i >= 2)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                           "mova    { z2.b, z3.b }, za0h.b[%w[za_index], 4:5] \n\t"
                           "fclamp  { z0.s-z3.s }, z12.s, z13.s \n\t"
                           "st1w    { z0.s, z1.s }, pn8, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           "st1w    { z2.s, z3.s }, pn8, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           : [dst_ptr] "+r"(dst_ptr)
                           : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
      za_index += 8;
      i -= 2;
    }
    if (i)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                           "fclamp  { z0.s, z1.s }, z12.s, z13.s \n\t"
                           "st1w    { z0.s, z1.s }, pn8, [%[dst_ptr]] \n\t"
                           :
                           : [za_index] "r"(za_index), [dst_ptr] "r"(dst_ptr), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
    }
  }

  // For sme_sgemm_NT_1xN_batch we use the store_tile function: we store the elements one by one - each in a
  // separate row, thus incrementing the destination pointer by destination matrix stride.
  template<unsigned TILE>
  static __attribute__ ((always_inline)) float* store_tile(float *dst_ptr,  sgemm_idx_t dst_stride_bytes, sgemm_idx_t limit) {
    register uint32_t za_index asm("w12") = 0;
    for ( ; za_index < limit; za_index += 4) {
      //  z12 = clamp_min,  z13 = clamp_max, p4 = the store predicate.
      __asm__ __volatile__ ("mova    { z0.s, z1.s, z2.s, z3.s }, za%[tile]h.s[%w[za_index], 0:3] \n\t"
                            "fclamp  { z0.s, z1.s, z2.s, z3.s }, z12.s, z13.s \n\t"
                            "st1w    { z0.s }, p4, [%[dst_ptr]]; \n\t"
                            "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                            : [dst_ptr] "+r" (dst_ptr)
                            : [za_index] "r" (za_index), [tile] "I" (TILE), [dst_stride_bytes] "r" (dst_stride_bytes)
                            : "memory");
      if (za_index+4 <= limit) {
        __asm__ __volatile__ ("st1w    { z1.s }, p4, [%[dst_ptr]]; \n\t"
                              "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                              "st1w    { z2.s }, p4, [%[dst_ptr]]; \n\t"
                              "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                              "st1w    { z3.s }, p4, [%[dst_ptr]]; \n\t"
                              "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                              : [dst_ptr] "+r" (dst_ptr)
                              : [dst_stride_bytes] "r" (dst_stride_bytes)
                              : "memory");
      }
      else {
        if (za_index + 1 < limit) {
          __asm__ __volatile__ ("st1w    { z1.s }, p4, [%[dst_ptr]]; \n\t"
                                "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                                : [dst_ptr] "+r" (dst_ptr)
                                : [dst_stride_bytes] "r" (dst_stride_bytes)
                                : "memory");
          if (za_index + 2 < limit) {
            __asm__ __volatile__ ("st1w    { z2.s }, p4, [%[dst_ptr]]; \n\t"
                                  "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                                  : [dst_ptr] "+r" (dst_ptr)
                                  : [dst_stride_bytes] "r" (dst_stride_bytes)
                                  : "memory");
          }
        }
      }
    }
    return dst_ptr;
  }

  __attribute__((noinline))
  static void sme_sgemm_NT_1xN_batch(const SME_Kernel_Params &params)
  {
    const float *lhs_ptr = params.lhs_ptr;
    const float *rhs_ptr = params.rhs_ptr;
          float *dst_ptr = params.dst_ptr;
    sgemm_idx_t lhs_stride_bytes = params.ruy_params->lhs_stride;
    sgemm_idx_t rhs_stride_bytes = params.ruy_params->rhs_stride;
    sgemm_idx_t dst_stride_bytes = params.ruy_params->dst_stride  << 2;
    sgemm_idx_t M_bytes = params.num_rows << 2;

    __asm__ __volatile__("zero    {za} \n\t"
                         "whilelt p4.b, xzr, %[M_bytes] \n\t"
                         :
                         : [M_bytes] "r"(M_bytes)
                         : "memory", "cc");

    uint32_t element_count = svcntw();
    if (params.num_cols > element_count * 2)
    {
      // 1x4
      __asm__ __volatile__("whilelt pn8.s, xzr, %[num_cols], vlx4 \n\t"
                           "pext    { p0.s, p1.s }, pn8[0] \n\t"
                           "pext    { p2.s, p3.s }, pn8[1] \n\t"
                           :
                           : [num_cols] "r"(params.num_cols)
                           : "memory", "cc");

      for (sgemm_idx_t k = 0; k < params.ruy_params->depth; ++k)
      {
        __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[rhs_ptr]] \n\t"
                             "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                             "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                             "fmopa   za1.s, p1/m, p4/m, z1.s, z4.s \n\t"
                             "fmopa   za2.s, p2/m, p4/m, z2.s, z4.s \n\t"
                             "fmopa   za3.s, p3/m, p4/m, z3.s, z4.s \n\t"
                             "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                             "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                             : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                             : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                             : "memory");
      }
      if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
      {
        const float *bias_ptr = NULL;
        if ((params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
        {
          bias_ptr = params.ruy_params->bias + params.start_col;
          __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[bias_ptr]] \n\t"
                               "dup      z4.s, %w[one_f] \n\t"
                               :
                               : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                               : "memory");
        }
        else
        {
          bias_ptr = params.ruy_params->bias + params.start_row;
          __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[bias_ptr]] \n\t"
                               "dup     z0.s, %w[one_f] \n\t"
                               "mov     z1.d, z0.d \n\t"
                               "mov     z2.d, z0.d \n\t"
                               "mov     z3.d, z0.d \n\t"
                               :
                               : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                               : "memory");
        }

        __asm__ __volatile__("fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                             "fmopa   za1.s, p1/m, p4/m, z1.s, z4.s \n\t"
                             "fmopa   za2.s, p2/m, p4/m, z2.s, z4.s \n\t"
                             "fmopa   za3.s, p3/m, p4/m, z3.s, z4.s \n\t"
                             : : : "memory");
      }
    }
    else if (params.num_cols > element_count)
    {
      // 1x2
      __asm__ __volatile__("whilelt pn8.s, xzr, %[num_cols], vlx2 \n\t"
                           "pext    { p0.s, p1.s }, pn8[0] \n\t"
                           :
                           : [num_cols] "r"(params.num_cols)
                           : "memory", "cc");
      if (params.ruy_params->depth >= SPLIT_THRESHOLD)
      {
        sgemm_idx_t k = params.ruy_params->depth;
        while (k >= 2)
        {
          __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[rhs_ptr]] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "fmopa   za1.s, p1/m, p4/m, z1.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               "ld1w    { z0.s, z1.s }, pn8/z, [%[rhs_ptr]] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "fmopa   za2.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "fmopa   za3.s, p1/m, p4/m, z1.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                               : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                               : "memory");
          k -= 2;
        }
        if (k)
        {
          __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[rhs_ptr]] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "fmopa   za1.s, p1/m, p4/m, z1.s, z4.s \n\t"
                               :
                               : [lhs_ptr] "r"(lhs_ptr), [rhs_ptr] "r"(rhs_ptr)
                               : "memory");
        }
        register uint32_t za_index asm("w11");
        for (za_index = 0; za_index < element_count; za_index += 4)
        {
          __asm__ __volatile__("mova    { z0.s, z1.s, z2.s, z3.s }, za.s[%w[za_index], 2] \n\t"
                               "mova    { z4.s, z5.s, z6.s, z7.s }, za.s[%w[za_index], 3] \n\t"
                               "fadd    za.s[%w[za_index], 0], { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "fadd    za.s[%w[za_index], 1], { z4.s, z5.s, z6.s, z7.s } \n\t"
                               :
                               : [za_index] "r"(za_index)
                               : "memory");
        }
      }
      else
      {
        for (sgemm_idx_t k = 0; k < params.ruy_params->depth; ++k)
        {
          __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[rhs_ptr]] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "fmopa   za1.s, p1/m, p4/m, z1.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                               : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                               : "memory");
        }
      }

      if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
      {
        const float *bias_ptr = NULL;
        if ((params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
        {
          bias_ptr = params.ruy_params->bias + params.start_col;
          __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[bias_ptr]] \n\t"
                               "dup      z4.s, %w[one_f] \n\t"
                               :
                               : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                               : "memory");
        }
        else
        {
          bias_ptr = params.ruy_params->bias + params.start_row;
          __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[bias_ptr]] \n\t"
                               "dup     z0.s, %w[one_f] \n\t"
                               "mov     z1.d, z0.d \n\t"
                               :
                               : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                               : "memory");
        }

        __asm__ __volatile__("fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                             "fmopa   za1.s, p1/m, p4/m, z1.s, z4.s \n\t"
                             : : : "memory");
      }
    }
    else
    {

      // 1x1
      __asm__ __volatile__("whilelt p0.s, xzr, %[num_cols] \n\t"
                           :
                           : [num_cols] "r"(params.num_cols)
                           : "memory", "cc");
      if (params.ruy_params->depth >= SPLIT_THRESHOLD)
      {
        sgemm_idx_t k = params.ruy_params->depth;
        while (k >= 4)
        {
          __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                               "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                               "fmopa   za1.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                               "fmopa   za2.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                               "fmopa   za3.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                               : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                               : "memory");
          k -= 4;
        }
        if (k--)
        {
          __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                               "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               :
                               : [lhs_ptr] "r"(lhs_ptr), [rhs_ptr] "r"(rhs_ptr)
                               : "memory");
          if (k--)
          {
            __asm__ __volatile__("add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                                 "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                                 "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                                 "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                                 "fmopa   za1.s, p0/m, p4/m, z0.s, z4.s \n\t"
                                 : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                                 : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                                 : "memory");
            if (k--)
            {
              __asm__ __volatile__("add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                                   "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                                   "ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                                   "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                                   "fmopa   za2.s, p0/m, p4/m, z0.s, z4.s \n\t"
                                   : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                                   : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                                   : "memory");
            }
          }
        }
        for (uint32_t input_tile = 1; input_tile < 4; ++input_tile)
        {
          register uint32_t za_index1 asm("w11");
          register uint32_t za_index2 asm("w10");

          for (za_index1 = 0; za_index1 < element_count; za_index1 += 4)
          {
            za_index2 = za_index1 + input_tile;
            __asm__ __volatile__("mova    { z0.s, z1.s, z2.s, z3.s }, za.s[%w[za_index2], 0] \n\t"
                                 "fadd    za.s[%w[za_index1], 0], { z0.s, z1.s, z2.s, z3.s } \n\t"
                                 :
                                 : [za_index1] "r"(za_index1), [za_index2] "r"(za_index2)
                                 : "memory");
          }
        }
      }
      else
      {
        for (sgemm_idx_t k = 0; k < params.ruy_params->depth; ++k)
        {
          __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[lhs_ptr]] \n\t"
                               "ld1w    { z0.s }, p0/z, [%[rhs_ptr]] \n\t"
                               "fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                               "add     %[lhs_ptr], %[lhs_ptr], %[lhs_stride_bytes] \n\t"
                               "add     %[rhs_ptr], %[rhs_ptr], %[rhs_stride_bytes] \n\t"
                               : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
                               : [lhs_stride_bytes] "r"(lhs_stride_bytes), [rhs_stride_bytes] "r"(rhs_stride_bytes)
                               : "memory");
        }
      }
      if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
      {
        const float *bias_ptr = NULL;
        if ((params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
        {
          bias_ptr = params.ruy_params->bias + params.start_col;
          __asm__ __volatile__("ld1w    { z0.s }, p0/z, [%[bias_ptr]] \n\t"
                               "dup      z4.s, %w[one_f] \n\t"
                               :
                               : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                               : "memory");
        }
        else
        {
          bias_ptr = params.ruy_params->bias + params.start_row;
          __asm__ __volatile__("ld1w    { z4.s }, p4/z, [%[bias_ptr]] \n\t"
                               "dup     z0.s, %w[one_f] \n\t"
                               :
                               : [bias_ptr] "r"(bias_ptr), [one_f] "r"(one_f)
                               : "memory");
        }

        __asm__ __volatile__("fmopa   za0.s, p0/m, p4/m, z0.s, z4.s \n\t"
                             : : : "memory");
      }
    }

     __asm__ __volatile__("dup z12.s, %w[clamp_min] \n\t"
                          "dup z13.s, %w[clamp_max] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max));
    
    uint32_t num_cols = params.num_cols;
    uint32_t tile0_limit = ARM64_SME_MIN(num_cols, element_count);
    dst_ptr = store_tile<0>(dst_ptr, dst_stride_bytes, tile0_limit);
    
    if (element_count < num_cols) {
      num_cols -= element_count;
      uint32_t tile1_limit = ARM64_SME_MIN(num_cols, element_count);
      dst_ptr = store_tile<1>(dst_ptr, dst_stride_bytes, tile1_limit);
      if (element_count < num_cols) {
        num_cols -= element_count;
        uint32_t tile2_limit = ARM64_SME_MIN(num_cols, element_count);
        dst_ptr = store_tile<2>(dst_ptr, dst_stride_bytes, tile2_limit);
        if (element_count < num_cols) {
          num_cols -= element_count;
          uint32_t tile3_limit = ARM64_SME_MIN(num_cols, element_count);
          dst_ptr = store_tile<3>(dst_ptr, dst_stride_bytes, tile3_limit);
        }
      }
    }
  }

  // The main function of the SME SGEMM. 
  static void sgemm_sme(const SME_RUY_Kernel_ParamsF32 &ruy_params,
                             sgemm_idx_t num_rows,sgemm_idx_t num_cols)
  {
    SME_Kernel_Params params;
    params.ruy_params = &ruy_params;
    const auto orig_start_row = ruy_params.start_row;
    const auto orig_start_col = ruy_params.start_col;
    
    const sgemm_idx_t element_count1 = svcntw();
    const sgemm_idx_t element_count2 = element_count1 * 2;
    const sgemm_idx_t element_count3 = element_count1 * 3;
    const sgemm_idx_t element_count4 = element_count1 * 4;
    sgemm_idx_t i = 0;

    // In case number of rows is big enough we use sme_sgemm_NT_4x1_batch.
    while (num_rows > i + element_count3)
    {
      sgemm_idx_t j = 0;
      for (; j < num_cols; j += element_count1)
      {
        params.num_rows = ARM64_SME_MIN(num_rows - i, element_count4);
        params.num_cols = ARM64_SME_MIN(num_cols - j, element_count1);
        params.lhs_ptr = params.ruy_params->lhs_base_ptr + i;
        params.rhs_ptr = params.ruy_params->rhs_base_ptr + j;
        params.dst_ptr = params.ruy_params->dst_base_ptr + params.ruy_params->dst_stride * j + i;
        params.start_row = orig_start_row + i;
        params.start_col = orig_start_col + j;

        sme_sgemm_NT_4x1_batch(params);
      }
      i += element_count4;
    }

    // Next we try to use sme_sgemm_NT_2x2_batch.
    while (num_rows > i + element_count1)
    {
      sgemm_idx_t j = 0;
      while (num_cols > j + element_count1)
      {
        params.num_rows = ARM64_SME_MIN(num_rows - i, element_count2);
        params.num_cols = ARM64_SME_MIN(num_cols - j, element_count2);
        params.lhs_ptr = params.ruy_params->lhs_base_ptr + i;
        params.rhs_ptr = params.ruy_params->rhs_base_ptr + j;
        params.dst_ptr = params.ruy_params->dst_base_ptr + params.ruy_params->dst_stride * j + i;
        params.start_row = orig_start_row + i;
        params.start_col = orig_start_col + j;

        sme_sgemm_NT_2x2_batch(params);
        j += element_count2;
      }

      // For last cols we use sme_sgemm_NT_2x1_batch.
      while (j < num_cols)
      {
        params.num_rows = ARM64_SME_MIN(num_rows - i, element_count2);
        params.num_cols = ARM64_SME_MIN(num_cols - j, element_count1);
        params.lhs_ptr = params.ruy_params->lhs_base_ptr + i;
        params.rhs_ptr = params.ruy_params->rhs_base_ptr + j;
        params.dst_ptr = &params.ruy_params->dst_base_ptr[params.ruy_params->dst_stride  * j + i];
        params.start_row = orig_start_row + i;
        params.start_col = orig_start_col + j;
        sme_sgemm_NT_2x1_batch(params);
        j += element_count1;
      }
      i += element_count2;
    }

    sgemm_idx_t block1_sz = element_count4;
    
    // The "leftover" case - handling last rows
    while (i < num_rows)
    {
      sgemm_idx_t j = 0;
      while (j < num_cols)
      {
        params.num_rows = ARM64_SME_MIN(num_rows - i, element_count1);
        params.num_cols = ARM64_SME_MIN(num_cols - j, block1_sz);
        params.lhs_ptr = params.ruy_params->lhs_base_ptr + i;
        params.rhs_ptr = params.ruy_params->rhs_base_ptr + j;
        params.dst_ptr = &params.ruy_params->dst_base_ptr[params.ruy_params->dst_stride  * j + i];
        params.start_row = orig_start_row + i;
        params.start_col = orig_start_col + j;
        sme_sgemm_NT_1xN_batch(params);
        j += block1_sz;
      }
      i += element_count1;
    }
  }

  // The entry function to the SGEMM SME kernel.
  void KernelFloatArm64SME(const SME_RUY_Kernel_ParamsF32 &params, int num_rows, int num_cols)
  {
    // We bound the SME code with SMSTART and SMSTOP.
    profiler::ScopeLabel label("Kernel (kArm64Sme)");
    uint8_t abi_stack[64] __attribute__((aligned(16)));
    SMSTART(abi_stack);
    sgemm_sme(params,num_rows,num_cols);
    SMSTOP(abi_stack);

    return;
  }

} // namespace ruy

#endif // RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)
