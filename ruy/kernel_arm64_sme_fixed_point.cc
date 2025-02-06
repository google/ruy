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

// Just like for floating point - when using SME for fixed point we logically "split" the processing 
// grid into 4 tiles. We process them using sme_gemmi8_NT_4x1_batch. Next we try sme_gemmi8_NT_2x2_batch, 
// and sme_gemmi8_NT_2x1_batch (depending on the destination height), and last sme_gemmi8_NT_1xN_batch.
// For i8 the GEMMi8 includes the rhs/lhs sums subtraction and adding the bias.
// In case the output is i16, i8 or u8 we down quantize the accumulators using the Z registers.
// This is done in the store functions.
// Per each "batch" GEMMi8 function we have a corresponding store function:
// 4x1: sme_store_dst_quads_i8 
// 2x2 and 2x1: sme_store_dst_singles_i8
// 1xN: sme_store_dst_singles_i8
// We have 3 variants for each store function - i8/u8 (templated), i16 and i32, since each has somehow different flow.

#if RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)
#include "ruy/arm64_sme_common.h"
namespace ruy
{
  static const gemmi8_idx_t minus_one = -1;
  struct SME_Kernel_ParamsI8
  {
    const SME_RUY_Kernel_Params8Bits *ruy_params;

    sgemm_idx_t num_rows;
    sgemm_idx_t num_cols;
    const void *lhs_ptr;
    const void *rhs_ptr;
    void *dst_ptr;
    std::int32_t start_row;
    std::int32_t start_col;
  };

  // As the output can be either i8, u8, i16 or i32 we need to increment the output pointer correctly.
  static bool getOutputNumBytes(const SME_RUY_Kernel_Params8Bits &ruy_params, int &num_bytes)
  {
    if ((ruy_params.dst_type_id == DstTypeId<std::int8_t>::kValue) ||
        (ruy_params.dst_type_id == DstTypeId<std::uint8_t>::kValue))
    {
      num_bytes = 1;
      return true;
    }
    else if (ruy_params.dst_type_id == DstTypeId<std::int16_t>::kValue)
    {
      num_bytes = 2;
      return true;
    }
    else if (ruy_params.dst_type_id == DstTypeId<std::int32_t>::kValue)
    {
      num_bytes = 4;
      return true;
    }

    RUY_DCHECK(false);
    return false;
  }

  // Used to increment the destination pointer correctly.
  inline sgemm_idx_t dst_idx(sgemm_idx_t i, sgemm_idx_t j, sgemm_idx_t LDC, size_t size_elem)
  {
    return (j * LDC + i) * size_elem;
  }

  template <unsigned TILE, bool SAME_EXP, bool isOutputSigned>
  FORCE_INLINE void *store_tile_i8(void *dst_ptr, gemmi8_idx_t dst_stride_bytes, gemmi8_idx_t limit,
                                   const std::int32_t *mult_ptr, const std::int32_t *exp_ptr)
  {
    // assumes z4 has multiplier, z8 exp,  z12 and z13 hold the clamp values, and z14 zero_point
    register uint32_t za_index asm("w12") = 0;
    for (; za_index < limit; za_index += 4)
    {
      __asm__ __volatile__("mova    { z0.s, z1.s, z2.s, z3.s }, za%[tile]h.s[%w[za_index], 0:3] " ::[za_index] "r"(za_index), [tile] "I"(TILE) : "memory");
      if (SAME_EXP)
      {
        __asm__ __volatile__("sqdmulh { z0.s, z1.s, z2.s, z3.s },  { z0.s, z1.s, z2.s, z3.s }, z4.s \n\t"
                             "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z8.s \n\t" ::: "memory");
      }
      else
      {
        __asm__ __volatile__("ld1rw   { z4.s }, p2/z, [%[mult_ptr]] \n\t"
                             "ld1rw   { z8.s }, p2/z, [%[exp_ptr]] \n\t"
                             "ld1rw   { z5.s }, p2/z, [%[mult_ptr], #4] \n\t"
                             "ld1rw   { z9.s }, p2/z, [%[exp_ptr], #4] \n\t"
                             "ld1rw   { z6.s }, p2/z, [%[mult_ptr], #8] \n\t"
                             "ld1rw   { z10.s }, p2/z, [%[exp_ptr], #8] \n\t"
                             "ld1rw   { z7.s }, p2/z, [%[mult_ptr], #12] \n\t"
                             "ld1rw   { z11.s }, p2/z, [%[exp_ptr], #12] \n\t"
                             "sqdmulh { z0.s, z1.s, z2.s, z3.s },  { z0.s, z1.s, z2.s, z3.s }, { z4.s, z5.s, z6.s, z7.s }  \n\t"
                             "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, { z8.s, z9.s, z10.s, z11.s } \n\t"
                             : : [mult_ptr] "r"(mult_ptr), [exp_ptr] "r"(exp_ptr) : "memory");
      }
      __asm__ __volatile__(
          "add     { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z14.s \n\t"
          :
          :
          : "memory");

      if (isOutputSigned)
      {
        __asm__ __volatile__("sqcvt   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                             "sclamp  z0.b, z12.b, z13.b \n\t"
                             :
                             :
                             : "memory");
      }
      else
      {
        __asm__ __volatile__("sqcvtu   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                             "uclamp  z0.b, z12.b, z13.b \n\t"
                             :
                             :
                             : "memory");
      }

      __asm__ __volatile__("st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           : [dst_ptr] "+r"(dst_ptr)
                           : [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
      if (za_index + 4 <= limit)
      {
        __asm__ __volatile__("splice  z0.b, p1, z0.b, z0.b \n\t"
                             "st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             "splice  z0.b, p1, z0.b, z0.b \n\t"
                             "st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             "splice  z0.b, p1, z0.b, z0.b \n\t"
                             "st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");
      }
      else
      {
        if (za_index + 1 < limit)
        {
          __asm__ __volatile__("splice  z0.b, p1, z0.b, z0.b \n\t"
                               "st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                               "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                               : [dst_ptr] "+r"(dst_ptr)
                               : [dst_stride_bytes] "r"(dst_stride_bytes)
                               : "memory");
          if (za_index + 2 < limit)
          {
            __asm__ __volatile__("splice  z0.b, p1, z0.b, z0.b \n\t"
                                 "st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                                 "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                                 : [dst_ptr] "+r"(dst_ptr)
                                 : [dst_stride_bytes] "r"(dst_stride_bytes)
                                 : "memory");
          }
        }
      }
      mult_ptr += 4;
      exp_ptr += 4;
    }
    return dst_ptr;
  }

  template <bool isOutputSigned>
  FORCE_INLINE void sme_store_dst_singles_i8(const SME_Kernel_ParamsI8 &params)
  {
    void *dst_ptr                 = params.dst_ptr;
    uint64_t elements_count       = svcntw();
    uint64_t za_rows              = elements_count * 4;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride;
    bool per_channel              = params.ruy_params->flags & RUY_ASM_FLAG_HAS_PERCHANNEL;
    bool per_row                  = !(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL);

    __asm__ __volatile__("whilelt p0.b, xzr, %[N_elts] \n\t"
                         "whilelt p1.b, %[elements_count], %[za_rows] \n\t"
                         "rev p1.b, p1.b \n\t"
                         : : [N_elts] "r"(params.num_rows), [elements_count] "r"(elements_count), [za_rows] "r"(za_rows) : "memory", "cc");

    __asm__ __volatile__("dup z12.b, %w[clamp_min] \n\t"
                         "dup z13.b, %w[clamp_max] \n\t"
                         "dup z14.s, %w[dst_zero_point] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max), [dst_zero_point] "r"(params.ruy_params->dst_zero_point));

    if (!per_channel || per_row)
    {
      if (per_channel)
      {
        const std::int32_t *multiplier = &params.ruy_params->multiplier_fixedpoint[params.start_row];
        const std::int32_t *exp = &params.ruy_params->multiplier_exponent[params.start_row];
        __asm__ __volatile__("whilelt p4.s, xzr, %[N_elts] \n\t"
                             "ld1w { z4.s }, p4/z,  [%[multiplier]]  \n\t"
                             "ld1w { z8.s }, p4/z, [%[exp]]  \n\t"
                             : : [multiplier] "r"(multiplier), [exp] "r"(exp), [N_elts] "r"(params.num_rows) : "memory", "cc");
      }
      else
      {
        std::int32_t multiplier = params.ruy_params->multiplier_fixedpoint[0];
        std::int32_t exp = params.ruy_params->multiplier_exponent[0];
        __asm__ __volatile__("dup z4.s,  %w[multiplier]  \n\t"
                             "dup z8.s,  %w[exp]  \n\t"
                             : : [multiplier] "r"(multiplier), [exp] "r"(exp));
      }
      uint32_t num_cols = params.num_cols;
      uint32_t tile0_limit = ARM64_SME_MIN(num_cols, elements_count);
      dst_ptr = store_tile_i8<0, true, isOutputSigned>(dst_ptr, dst_stride_bytes, tile0_limit, nullptr, nullptr);
      if (elements_count < num_cols)
      {
        num_cols -= elements_count;
        uint32_t tile1_limit = ARM64_SME_MIN(num_cols, elements_count);
        dst_ptr = store_tile_i8<1, true, isOutputSigned>(dst_ptr, dst_stride_bytes, tile1_limit, nullptr, nullptr);
        if (elements_count < num_cols)
        {
          num_cols -= elements_count;
          uint32_t tile2_limit = ARM64_SME_MIN(num_cols, elements_count);
          dst_ptr = store_tile_i8<2, true, isOutputSigned>(dst_ptr, dst_stride_bytes, tile2_limit, nullptr, nullptr);
          if (elements_count < num_cols)
          {
            num_cols -= elements_count;
            uint32_t tile3_limit = ARM64_SME_MIN(num_cols, elements_count);
            dst_ptr = store_tile_i8<3, true, isOutputSigned>(dst_ptr, dst_stride_bytes, tile3_limit, nullptr, nullptr);
          }
        }
      }
    }
    else
    {
      __asm__ __volatile__("ptrue p2.s" : : : "memory", "cc");
      const std::int32_t *mult_ptr = &params.ruy_params->multiplier_fixedpoint[params.start_col];
      const std::int32_t *exp_ptr = &params.ruy_params->multiplier_exponent[params.start_col];

      uint32_t num_cols = params.num_cols;
      uint32_t tile0_limit = ARM64_SME_MIN(num_cols, elements_count);
      dst_ptr = store_tile_i8<0, false, isOutputSigned>(dst_ptr, dst_stride_bytes, tile0_limit, mult_ptr, exp_ptr);
      if (elements_count < num_cols)
      {
        num_cols -= elements_count;
        mult_ptr += elements_count;
        exp_ptr += elements_count;
        uint32_t tile1_limit = ARM64_SME_MIN(num_cols, elements_count);
        dst_ptr = store_tile_i8<1, false, isOutputSigned>(dst_ptr, dst_stride_bytes, tile1_limit, mult_ptr, exp_ptr);
        if (elements_count < num_cols)
        {
          num_cols -= elements_count;
          mult_ptr += elements_count;
          exp_ptr += elements_count;
          uint32_t tile2_limit = ARM64_SME_MIN(num_cols, elements_count);
          dst_ptr = store_tile_i8<2, false, isOutputSigned>(dst_ptr, dst_stride_bytes, tile2_limit, mult_ptr, exp_ptr);
          if (elements_count < num_cols)
          {
            num_cols -= elements_count;
            mult_ptr += elements_count;
            exp_ptr += elements_count;
            uint32_t tile3_limit = ARM64_SME_MIN(num_cols, elements_count);
            dst_ptr = store_tile_i8<3, false, isOutputSigned>(dst_ptr, dst_stride_bytes, tile3_limit, mult_ptr, exp_ptr);
          }
        }
      }
    }
  }

  template <bool isOutputSigned>
  FORCE_INLINE void sme_store_dst_pairs_i8(const SME_Kernel_ParamsI8 &params)
  {
    void *dst_ptr                 = params.dst_ptr;
    uint64_t za_rows              = svcntb();
    gemmi8_idx_t i                = params.num_cols;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride;
    bool per_channel              = params.ruy_params->flags & RUY_ASM_FLAG_HAS_PERCHANNEL;
    bool per_row                  = !(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL);

    __asm__ __volatile__("whilelt p0.b, xzr, %[N_elts]" : : [N_elts] "r"(params.num_rows) : "memory", "cc");

    __asm__ __volatile__("dup z12.b, %w[clamp_min] \n\t"
                         "dup z13.b, %w[clamp_max] \n\t"
                         "dup z14.h, %w[dst_zero_point] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max), [dst_zero_point] "r"(params.ruy_params->dst_zero_point));

    if (per_channel && per_row)
    {
      const std::int32_t *multiplier = &params.ruy_params->multiplier_fixedpoint[params.start_row];
      const std::int32_t *exp = &params.ruy_params->multiplier_exponent[params.start_row];
      __asm__ __volatile__("ld1w { z4.s, z5.s }, pn8/z,  [%[multiplier]]  \n\t"
                           "ld1w { z8.s, z9.s }, pn8/z, [%[exp]]  \n\t"
                           : : [multiplier] "r"(multiplier), [exp] "r"(exp));
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                             "sqdmulh { z0.s, z1.s },  { z0.s, z1.s }, {z4.s, z5.s} \n\t"
                             "srshl   { z0.s, z1.s }, { z0.s, z1.s }, {z8.s, z9.s } \n\t"
                             "sqcvt   z0.h, { z0.s, z1.s } \n\t"
                             "sqadd     z0.h, z0.h, z14.h \n\t"
                             :
                             : [za_index] "r"(za_index)
                             : "memory");

        if (isOutputSigned)
        {
          __asm__ __volatile__("sqxtnb  z0.b, z0.h \n\t"
                               "uzp1 z0.b, z0.b, z0.b \n\t"
                               "sclamp  z0.b, z12.b, z13.b \n\t"
                               :
                               :
                               : "memory");
        }
        else
        {
          __asm__ __volatile__("sqxtunb  z0.b, z0.h \n\t"
                               "uzp1 z0.b, z0.b, z0.b \n\t"
                               "uclamp  z0.b, z12.b, z13.b \n\t"
                               :
                               :
                               : "memory");
        }

        __asm__ __volatile__("st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");

        za_index += 4;
        if (za_index >= za_rows)
        {
          za_index = 2;
        }
        --i;
      }
    }
    else if (per_channel && !per_row)
    {
      const std::int32_t *mult_ptr = &params.ruy_params->multiplier_fixedpoint[params.start_col];
      const std::int32_t *exp_ptr = &params.ruy_params->multiplier_exponent[params.start_col];
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        std::int32_t multiplier = *(mult_ptr++);
        std::int32_t exponent = *(exp_ptr++);
        __asm__ __volatile__(
            "mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
            "dup   z10.s, %w[multiplier] \n\t"
            "dup   z11.s, %w[exponent] \n\t"
            "sqdmulh { z0.s, z1.s },  { z0.s, z1.s }, z10.s \n\t"
            "srshl   { z0.s, z1.s }, { z0.s, z1.s }, z11.s \n\t"
            "sqcvt   z0.h, { z0.s, z1.s } \n\t"
            "sqadd   z0.h, z0.h, z14.h \n\t"
            :
            : [za_index] "r"(za_index), [multiplier] "r"(multiplier), [exponent] "r"(exponent)
            : "memory");

        if (isOutputSigned)
        {
          __asm__ __volatile__("sqxtnb  z0.b, z0.h \n\t"
                               "uzp1 z0.b, z0.b, z0.b \n\t"
                               "sclamp  z0.b, z12.b, z13.b \n\t"
                               :
                               :
                               : "memory");
        }
        else
        {
          __asm__ __volatile__("sqxtunb  z0.b, z0.h \n\t"
                               "uzp1 z0.b, z0.b, z0.b \n\t"
                               "uclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }

        __asm__ __volatile__("st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");

        za_index += 4;
        if (za_index >= za_rows)
        {
          za_index = 2;
        }
        --i;
      }
    }
    else
    {
      std::int32_t multiplier = params.ruy_params->multiplier_fixedpoint[0];
      std::int32_t exp = params.ruy_params->multiplier_exponent[0];

      __asm__ __volatile__("dup z10.s,  %w[multiplier]  \n\t"
                           "dup z11.s,  %w[exp]  \n\t"
                           : : [multiplier] "r"(multiplier), [exp] "r"(exp));
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                             "sqdmulh { z0.s, z1.s },  { z0.s, z1.s }, z10.s \n\t"
                             "srshl   { z0.s, z1.s }, { z0.s, z1.s }, z11.s \n\t"
                             "sqcvt   z0.h, { z0.s, z1.s } \n\t"
                             "sqadd     z0.h, z0.h, z14.h \n\t"
                             :
                             : [za_index] "r"(za_index)
                             : "memory");

        if (isOutputSigned)
        {
          __asm__ __volatile__("sqxtnb  z0.b, z0.h \n\t"
                               "uzp1 z0.b, z0.b, z0.b \n\t"
                               "sclamp  z0.b, z12.b, z13.b \n\t"
                               :
                               :
                               : "memory");
        }
        else
        {
          __asm__ __volatile__("sqxtunb  z0.b, z0.h \n\t"
                               "uzp1 z0.b, z0.b, z0.b \n\t"
                               "uclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }

        __asm__ __volatile__("st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");
        za_index += 4;
        if (za_index >= za_rows)
        {
          za_index = 2;
        }
        --i;
      }
    }
  }

/* 
  In the store_tile function we store the elements one by one - each in a separate row, 
 thus incrementing the destination pointer by destination matrix stride.
  The function implements the next steps:
     1. Moving the accumulators (za) content to z registers.
     2. Applying the fixed point multiplier.
     3. Adding the destination zero point.
     4. Conversion to destination 8-bits.
     5. Clamping the result.
     6. Storing the final result into the destination matrix.
*/

  template <bool isOutputSigned>
  FORCE_INLINE void sme_store_dst_quads_i8(const SME_Kernel_ParamsI8 &params)
  {
    void *dst_ptr                 = params.dst_ptr;
    gemmi8_idx_t i                = params.num_cols;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride;
    bool per_channel              = params.ruy_params->flags & RUY_ASM_FLAG_HAS_PERCHANNEL;
    bool per_row                  = !(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL);

    __asm__ __volatile__("whilelt p0.b, xzr, %[N_elts] \n\t"
                         "dup z12.b, %w[clamp_min] \n\t"
                         "dup z13.b, %w[clamp_max] \n\t"
                         "dup z14.s, %w[dst_zero_point] \n\t"
                         :
                         : [N_elts] "r"(params.num_rows), [clamp_min] "r"(params.ruy_params->clamp_min),
                           [clamp_max] "r"(params.ruy_params->clamp_max), [dst_zero_point] "r"(params.ruy_params->dst_zero_point)
                         : "memory", "cc");

    if (per_channel && per_row)
    {
      const std::int32_t *multiplier = &params.ruy_params->multiplier_fixedpoint[params.start_row];
      const std::int32_t *exp = &params.ruy_params->multiplier_exponent[params.start_row];
      
      __asm__ __volatile__("ld1w { z4.s, z5.s, z6.s, z7.s }, pn8/z,  [%[multiplier]]  \n\t"
                           "ld1w { z8.s, z9.s, z10.s, z11.s }, pn8/z, [%[exp]]  \n\t"
                           : : [multiplier] "r"(multiplier), [exp] "r"(exp));
      
      register uint32_t za_index asm("w12") = 0;
      
      while (i)
      {
        __asm__ __volatile__("mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
                             "sqdmulh { z0.s, z1.s, z2.s, z3.s },  { z0.s, z1.s, z2.s, z3.s }, {z4.s, z5.s, z6.s, z7.s} \n\t"
                             "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, {z8.s, z9.s, z10.s, z11.s} \n\t"
                             "add     { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z14.s \n\t"
                             :
                             : [za_index] "r"(za_index)
                             : "memory");

        if (isOutputSigned)
        {
          __asm__ __volatile__("sqcvt   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "sclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }
        else
        {
          __asm__ __volatile__("sqcvtu   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "uclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }

        __asm__ __volatile__("st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");
        za_index += 4;
        --i;
      }
    }
    else if (per_channel && !per_row)
    {
      const std::int32_t *mult_ptr = &params.ruy_params->multiplier_fixedpoint[params.start_col];
      const std::int32_t *exp_ptr = &params.ruy_params->multiplier_exponent[params.start_col];
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        std::int32_t multiplier = *(mult_ptr++);
        std::int32_t exponent = *(exp_ptr++);
        __asm__ __volatile__(
            "mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
            "dup   z10.s, %w[multiplier] \n\t"
            "dup   z11.s, %w[exponent] \n\t"
            "sqdmulh { z0.s, z1.s, z2.s, z3.s },  { z0.s, z1.s, z2.s, z3.s }, z10.s \n\t"
            "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z11.s \n\t"
            "add     { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z14.s \n\t"
            :
            : [za_index] "r"(za_index), [multiplier] "r"(multiplier), [exponent] "r"(exponent)
            : "memory");

        if (isOutputSigned)
        {
          __asm__ __volatile__("sqcvt   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "sclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }
        else
        {
          __asm__ __volatile__("sqcvtu   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "uclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }

        __asm__ __volatile__("st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");

        za_index += 4;
        --i;
      }
    }
    else
    {
      std::int32_t multiplier = params.ruy_params->multiplier_fixedpoint[0];
      std::int32_t exp = params.ruy_params->multiplier_exponent[0];

      __asm__ __volatile__("dup z10.s,  %w[multiplier]  \n\t"
                           "dup z11.s,  %w[exp]  \n\t"
                           : : [multiplier] "r"(multiplier), [exp] "r"(exp));
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__(
            "mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
            "sqdmulh { z0.s, z1.s, z2.s, z3.s },  { z0.s, z1.s, z2.s, z3.s }, z10.s \n\t"
            "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z11.s \n\t"
            "add     { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z14.s \n\t"
            :
            : [za_index] "r"(za_index)
            : "memory");

        if (isOutputSigned)
        {
          __asm__ __volatile__("sqcvt   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "sclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }
        else
        {
          __asm__ __volatile__("sqcvtu   z0.b, { z0.s, z1.s, z2.s, z3.s } \n\t"
                               "uclamp  z0.b, z12.b, z13.b \n\t"
                               : : : "memory");
        }

        __asm__ __volatile__(
            "st1b    { z0.b }, p0, [%[dst_ptr]] \n\t"
            "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
            : [dst_ptr] "+r"(dst_ptr)
            : [dst_stride_bytes] "r"(dst_stride_bytes)
            : "memory");

        za_index += 4;
        --i;
      }
    }
  }

/* 
  In the store_tile function we store the elements one by one - each in a separate row (thus incrementing the dst ptr by its stride).
     The function implements the next steps:
     1. Moving the accumulators (za) content to z registers.
     2. Applying the fixed point multiplier: shift left, multiply (+ immediate shift), shift right.
     3. Adding the destination zero point.
     4. Conversion to 16 bits.
     5. Clamping the result.
     6. Storing the final result into the destination matrix.
*/
  template <unsigned TILE, bool SAME_EXP>
  FORCE_INLINE void *store_tile_i16(void *dst_ptr, gemmi8_idx_t dst_stride_bytes, gemmi8_idx_t limit,
                                    const std::int32_t *mult_ptr, const std::int32_t *exp_ptr)
  {
    register uint32_t za_index asm("w12") = 0;

    for (; za_index < limit; za_index += 4)
    {
      // We use the z20-23 registers to load the accumulators as we cannot use the high registers for the smin and sub calls.
      __asm__ __volatile__("mova    { z20.s, z21.s, z22.s, z23.s }, za%[tile]h.s[%w[za_index], 0:3] " ::[za_index] "r"(za_index), [tile] "I"(TILE) : "memory");
      if (SAME_EXP)
      { 
        // z8 = shift left, z4 = multiplier, z15 = shift right (negative value)
        __asm__ __volatile__("srshl   { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, z8.s \n\t"
                             "sqdmulh { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, z4.s \n\t"
                             "srshl   { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, z15.s \n\t"
                             : : : "memory");
      }
      else
      {
        __asm__ __volatile__("ld1rw   { z4.s }, p2/z, [%[mult_ptr]] \n\t"
                             "ld1rw   { z8.s }, p2/z, [%[exp_ptr]] \n\t"
                             "ld1rw   { z5.s }, p2/z, [%[mult_ptr], #4] \n\t"
                             "ld1rw   { z9.s }, p2/z, [%[exp_ptr], #4] \n\t"
                             "ld1rw   { z6.s }, p2/z, [%[mult_ptr], #8] \n\t"
                             "ld1rw   { z10.s }, p2/z, [%[exp_ptr], #8] \n\t"
                             "ld1rw   { z7.s }, p2/z, [%[mult_ptr], #12] \n\t"
                             "ld1rw   { z11.s }, p2/z, [%[exp_ptr], #12] \n\t"
                             "dup z0.s, %w[minus_one] \n\t"
                             "dup z1.s, %w[minus_one] \n\t"
                             "dup z2.s, %w[minus_one] \n\t"
                             "dup z3.s, %w[minus_one] \n\t"
                             "smin z0.s, p7/m, z0.s, z8.s  \n\t"
                             "smin z1.s, p7/m, z1.s, z9.s  \n\t"
                             "smin z2.s, p7/m, z2.s, z10.s \n\t"
                             "smin z3.s, p7/m, z3.s, z11.s \n\t"
                             "sub  z8.s,  p7/m, z8.s,  z0.s \n\t"
                             "sub  z9.s,  p7/m, z9.s,  z1.s \n\t"
                             "sub  z10.s, p7/m, z10.s, z2.s \n\t"
                             "sub  z11.s, p7/m, z11.s, z3.s \n\t"
                             : : [mult_ptr] "r"(mult_ptr), [exp_ptr] "r"(exp_ptr), [minus_one] "r"(minus_one) : "memory");
        
        __asm__ __volatile__("srshl   { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, { z8.s, z9.s, z10.s, z11.s } \n\t"
                             "sqdmulh { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, { z4.s, z5.s, z6.s, z7.s }  \n\t"
                             "srshl   { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, { z0.s, z1.s, z2.s, z3.s } \n\t"
                             : : [mult_ptr] "r"(mult_ptr), [exp_ptr] "r"(exp_ptr), [minus_one] "r"(minus_one) : "memory");
      }
      // z12 = clamp min, z13 = clamp max, z14 = dst zero_point
      __asm__ __volatile__("add     { z20.s, z21.s, z22.s, z23.s }, { z20.s, z21.s, z22.s, z23.s }, z14.s \n\t"
                           "sqcvt   z0.h, { z20.s, z21.s} \n\t"
                           "sqcvt   z1.h, { z22.s, z23.s} \n\t"
                           "sclamp  {z0.h, z1.h}, z12.h, z13.h \n\t"
                          " st1h    {z0.h}, p0, [%[dst_ptr]] \n\t"
                          " add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                          : [dst_ptr] "+r"(dst_ptr)
                          : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                          : "memory");

     if (za_index + 4 <= limit)
     {
       __asm__ __volatile__("splice z0.h, p1, z0.h, z0.h \n\t"
                            "st1h {z0.h}, p0, [%[dst_ptr]] \n\t"
                            "add %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                            "st1h {z1.h}, p0, [%[dst_ptr]] \n\t "
                            "add %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                            "splice z1.h, p1, z1.h, z1.h \n\t"
                            "st1h {z1.h}, p0, [%[dst_ptr]] \n\t"
                            "add %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                            : [dst_ptr] "+r"(dst_ptr)
                            : [dst_stride_bytes] "r"(dst_stride_bytes)
                            : "memory");
      }
      else
      {
        if (za_index + 1 < limit)
        {
          __asm__ __volatile__("splice z0.h, p1, z0.h, z0.h \n\t"
                               "st1h {z0.h} , p0, [%[dst_ptr]] \n\t"
                               "add %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                               : [dst_ptr] "+r"(dst_ptr)
                               : [dst_stride_bytes] "r"(dst_stride_bytes)
                               : "memory");
          if (za_index + 2 < limit)
          {
            __asm__ __volatile__( // note No splice
                "st1h {z1.h}, p0, [%[dst_ptr]] \n\t"
                "add %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                : [dst_ptr] "+r"(dst_ptr)
                : [dst_stride_bytes] "r"(dst_stride_bytes)
                : "memory");
          }
        }
      }
      mult_ptr += 4;
      exp_ptr += 4;
    }
    return dst_ptr; 
  }

  FORCE_INLINE void sme_store_dst_singles_i16(const SME_Kernel_ParamsI8 &params)
  {
    void *dst_ptr                 = params.dst_ptr;
    uint64_t elements_count       = svcntw();
    uint64_t elements_count_h     = elements_count * 2;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride << 1;
    bool per_channel              = params.ruy_params->flags & RUY_ASM_FLAG_HAS_PERCHANNEL;
    bool per_row                  = !(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL);

    __asm__ __volatile__("whilelt p0.h, xzr, %[N_elts] \n\t"
                         "whilelt p1.h, %[elements_count], %[elements_count_h] \n\t"
                         "rev p1.h, p1.h \n\t" 
                         "ptrue p7.h \n\t"
                         "dup z12.h, %w[clamp_min] \n\t"
                         "dup z13.h, %w[clamp_max] \n\t"
                         "dup z14.s, %w[dst_zero_point] \n\t"
                         :
                         : [N_elts] "r"(params.num_rows), [elements_count] "r"(elements_count), [elements_count_h] "r"(elements_count_h),
                           [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max),
                           [dst_zero_point] "r"(params.ruy_params->dst_zero_point)
                         : "memory", "cc");

    if (!per_channel || per_row)
    {
      if (per_channel)
      {
        const std::int32_t *multiplier = &params.ruy_params->multiplier_fixedpoint[params.start_row];
        const std::int32_t *exp = &params.ruy_params->multiplier_exponent[params.start_row];
        __asm__ __volatile__("whilelt p4.s, xzr, %[N_elts] \n\t"
                             "ld1w { z4.s }, p4/z,  [%[multiplier]]  \n\t"
                             "ld1w { z8.s }, p4/z, [%[exp]]  \n\t"
                             "dup z15.s, %w[minus_one] \n\t"
                             "smin  z15.s, p7/m, z15.s, z8.s \n\t"
                             "sub  z8.s, p7/m, z8.s, z15.s \n\t"
                             : : [multiplier] "r"(multiplier), [exp] "r"(exp), [N_elts] "r"(params.num_rows), [minus_one] "r"(minus_one)
                             : "memory", "cc");
      }
      else
      {
        std::int32_t multiplier = params.ruy_params->multiplier_fixedpoint[0];
        std::int32_t exp = params.ruy_params->multiplier_exponent[0];
        std::int32_t shr = ARM64_SME_MIN(-1, exp);
        std::int32_t shl = exp - shr;

        __asm__ __volatile__("dup z4.s,  %w[multiplier]  \n\t"
                             "dup z8.s,  %w[shl]  \n\t"
                             "dup z15.s, %w[shr] \n\t"
                             : : [multiplier] "r"(multiplier), [shl] "r"(shl),[shr] "r"(shr));
      }
      
      uint32_t num_cols = params.num_cols;
      uint32_t tile0_limit = ARM64_SME_MIN(num_cols, elements_count);
      dst_ptr = store_tile_i16<0, true>(dst_ptr, dst_stride_bytes, tile0_limit, nullptr, nullptr);
      if (elements_count < num_cols)
      {
        num_cols -= elements_count;
        uint32_t tile1_limit = ARM64_SME_MIN(num_cols, elements_count);
        dst_ptr = store_tile_i16<1, true>(dst_ptr, dst_stride_bytes, tile1_limit, nullptr, nullptr);
        if (elements_count < num_cols)
        {
          num_cols -= elements_count;
          uint32_t tile2_limit = ARM64_SME_MIN(num_cols, elements_count);
          dst_ptr = store_tile_i16<2, true>(dst_ptr, dst_stride_bytes, tile2_limit, nullptr, nullptr);
          if (elements_count < num_cols)
          {
            num_cols -= elements_count;
            uint32_t tile3_limit = ARM64_SME_MIN(num_cols, elements_count);
            dst_ptr = store_tile_i16<3, true>(dst_ptr, dst_stride_bytes, tile3_limit, nullptr, nullptr);
          }
        }
      }
    }
    else
    {
      __asm__ __volatile__("ptrue p2.s" : : : "memory", "cc");
      const std::int32_t *mult_ptr = &params.ruy_params->multiplier_fixedpoint[params.start_col];
      const std::int32_t *exp_ptr = &params.ruy_params->multiplier_exponent[params.start_col];

      uint32_t num_cols = params.num_cols;
      uint32_t tile0_limit = ARM64_SME_MIN(num_cols, elements_count);
      dst_ptr = store_tile_i16<0, false>(dst_ptr, dst_stride_bytes, tile0_limit, mult_ptr, exp_ptr);
      if (elements_count < num_cols)
      {
        num_cols -= elements_count;
        mult_ptr += elements_count;
        exp_ptr += elements_count;
        uint32_t tile1_limit = ARM64_SME_MIN(num_cols, elements_count);
        dst_ptr = store_tile_i16<1, false>(dst_ptr, dst_stride_bytes, tile1_limit, mult_ptr, exp_ptr);
        if (elements_count < num_cols)
        {
          mult_ptr += elements_count;
          exp_ptr += elements_count;
          num_cols -= elements_count;
          uint32_t tile2_limit = ARM64_SME_MIN(num_cols, elements_count);
          dst_ptr = store_tile_i16<2, false>(dst_ptr, dst_stride_bytes, tile2_limit, mult_ptr, exp_ptr);
          if (elements_count < num_cols)
          {
            num_cols -= elements_count;
            mult_ptr += elements_count;
            exp_ptr += elements_count;
            uint32_t tile3_limit = ARM64_SME_MIN(num_cols, elements_count);
            dst_ptr = store_tile_i16<3, false>(dst_ptr, dst_stride_bytes, tile3_limit, mult_ptr, exp_ptr);
          }
        }
      }
    }
  }

  FORCE_INLINE void sme_store_dst_pairs_i16(const SME_Kernel_ParamsI8 &params)
  {
    void *dst_ptr                 = params.dst_ptr;
    uint64_t za_rows              = svcntb();
    gemmi8_idx_t i                = params.num_cols;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride << 1;
    bool per_channel              = params.ruy_params->flags & RUY_ASM_FLAG_HAS_PERCHANNEL;
    bool per_row                  = !(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL);

    __asm__ __volatile__("whilelt p0.h, xzr, %[N_elts] \n\t"
                         "ptrue p7.h \n\t"
                         : : [N_elts] "r"(params.num_rows) : "memory", "cc");

    __asm__ __volatile__("dup z12.h, %w[clamp_min] \n\t"
                         "dup z13.h, %w[clamp_max] \n\t"
                         "dup z14.s, %w[dst_zero_point] \n\t"
                         "dup z15.s, %w[minus_one] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max),
                             [dst_zero_point] "r"(params.ruy_params->dst_zero_point), [minus_one] "r"(minus_one));

    if (per_channel && per_row)
    {
      const std::int32_t *multiplier = &params.ruy_params->multiplier_fixedpoint[params.start_row];
      const std::int32_t *exp = &params.ruy_params->multiplier_exponent[params.start_row];
      __asm__ __volatile__("ld1w { z4.s, z5.s },  pn8/z, [%[multiplier]]  \n\t"
                           "ld1w { z8.s, z9.s },  pn8/z, [%[exp]]  \n\t"
                           "ld1w { z10.s, z11.s}, pn8/z, [%[exp]]  \n\t"
                           "smin { z8.s, z9.s },  { z8.s, z9.s }, z15.s\n\t"
                           "sub  z10.s, z10.s, z8.s  \n\t"
                           "sub  z11.s, z11.s, z9.s  \n\t"
                           : : [multiplier] "r"(multiplier), [exp] "r"(exp));

      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                             "srshl   { z0.s, z1.s }, { z0.s, z1.s }, { z10.s, z11.s } \n\t"
                             "sqdmulh { z0.s, z1.s }, { z0.s, z1.s }, { z4.s, z5.s } \n\t"
                             "srshl   { z0.s, z1.s }, { z0.s, z1.s }, { z8.s, z9.s } \n\t"
                             "add     { z0.s, z1.s }, { z0.s, z1.s }, z14.s \n\t"
                             "sqcvt   z0.h, { z0.s, z1.s } \n\t"
                             "sclamp  z0.h, z12.h, z13.h \n\t"
                             "st1h    { z0.h }, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");

        za_index += 4;
        if (za_index >= za_rows)
        {
          za_index = 2;
        }
        --i;
      }
    }
    else if (per_channel && !per_row)
    {
      const std::int32_t *mult_ptr  = &params.ruy_params->multiplier_fixedpoint[params.start_col];
      const std::int32_t *exp_ptr   = &params.ruy_params->multiplier_exponent[params.start_col];
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        std::int32_t multiplier = *(mult_ptr++);
        std::int32_t exponent = *(exp_ptr++);
        __asm__ __volatile__(
            "mova     { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
            "dup      z10.s, %w[multiplier] \n\t"
            "dup      z11.s, %w[exponent] \n\t"
            "dup      z15.s, %w[minus_one] \n\t"
            "smin     z15.s, p7/m, z15.s, z11.s \n\t"
            "sub      z11.s, z11.s, z15.s \n\t"
            "srshl    { z0.s, z1.s }, { z0.s, z1.s }, z11.s \n\t"
            "sqdmulh  { z0.s, z1.s }, { z0.s, z1.s }, z10.s \n\t"
            "srshl    { z0.s, z1.s }, { z0.s, z1.s }, z15.s \n\t"
            "add      { z0.s, z1.s }, { z0.s, z1.s }, z14.s \n\t"
            "sqcvt    z0.h, { z0.s, z1.s } \n\t"
            "sclamp   z0.h, z12.h, z13.h \n\t"
            "st1h     { z0.h }, p0, [%[dst_ptr]] \n\t"
            "add      %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
            : [dst_ptr] "+r"(dst_ptr)
            : [za_index] "r"(za_index), [multiplier] "r"(multiplier), [exponent] "r"(exponent),
              [dst_stride_bytes] "r"(dst_stride_bytes), [minus_one] "r"(minus_one)
            : "memory");

        za_index += 4;
        if (za_index >= za_rows)
        {
          za_index = 2;
        }
        --i;
      }
    }
    else
    {
      std::int32_t multiplier = params.ruy_params->multiplier_fixedpoint[0];
      std::int32_t exp = params.ruy_params->multiplier_exponent[0];
      std::int32_t shr = ARM64_SME_MIN(-1, exp);
      std::int32_t shl = exp - shr;

      __asm__ __volatile__("dup  z10.s, %w[multiplier]  \n\t"
                           "dup  z11.s, %w[shl] \n\t"
                           "dup  z15.s, %w[shr] \n\t"
                           :
                           : [multiplier] "r"(multiplier), [shl] "r"(shl), [shr] "r"(shr)
                           : "memory");

      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                             "srshl   { z0.s, z1.s }, { z0.s, z1.s }, z11.s \n\t"
                             "sqdmulh { z0.s, z1.s }, { z0.s, z1.s }, z10.s \n\t"
                             "srshl   { z0.s, z1.s }, { z0.s, z1.s }, z15.s \n\t"
                             "add   { z0.s, z1.s }, { z0.s, z1.s }, z14.s \n\t"
                             "sqcvt   z0.h, { z0.s, z1.s } \n\t"
                             "sclamp  z0.h, z12.h, z13.h \n\t"
                             "st1h    {z0.h}, p0, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");

        za_index += 4;
        if (za_index >= za_rows)
        {
          za_index = 2;
        }
        --i;
      }
    }
  }

  FORCE_INLINE void sme_store_dst_quads_i16(const SME_Kernel_ParamsI8 &params)
  {
    void *dst_ptr                 = params.dst_ptr;
    gemmi8_idx_t i                = params.num_cols;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride << 1;
    bool per_channel              = params.ruy_params->flags & RUY_ASM_FLAG_HAS_PERCHANNEL;
    bool per_row                  = !(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL);

    __asm__ __volatile__("whilelt pn9.h, xzr, %[N_elts], vlx2 \n\t"
                         "ptrue p7.h \n\t"
                         : : [N_elts] "r"(params.num_rows) : "memory", "cc");

    __asm__ __volatile__("dup z12.h, %w[clamp_min] \n\t"
                         "dup z13.h, %w[clamp_max] \n\t"
                         "dup z14.s, %w[dst_zero_point] \n\t"
                         "dup z15.s, %w[minus_one] \n\t"
                         : : [clamp_min] "r"(params.ruy_params->clamp_min), [clamp_max] "r"(params.ruy_params->clamp_max),
                             [dst_zero_point] "r"(params.ruy_params->dst_zero_point), [minus_one] "r"(minus_one));

    if (per_channel && per_row)
    {
      const std::int32_t *multiplier = &params.ruy_params->multiplier_fixedpoint[params.start_row];
      const std::int32_t *exp = &params.ruy_params->multiplier_exponent[params.start_row];
      __asm__ __volatile__("ld1w { z4.s, z5.s, z6.s, z7.s }, pn8/z,  [%[multiplier]]  \n\t"
                           "ld1w { z8.s, z9.s, z10.s, z11.s }, pn8/z, [%[exp]]  \n\t"
                           "ld1w { z16.s, z17.s, z18.s, z19.s }, pn8/z, [%[exp]]  \n\t"
                           "smin { z8.s, z9.s, z10.s, z11.s }, { z8.s, z9.s, z10.s, z11.s }, z15.s\n\t"
                           "sub  z16.s, z16.s, z8.s  \n\t"
                           "sub  z17.s, z17.s, z9.s  \n\t"
                           "sub  z18.s, z18.s, z10.s  \n\t"
                           "sub  z19.s, z19.s, z11.s  \n\t"
                           : : [multiplier] "r"(multiplier), [exp] "r"(exp));

      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__("mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
                             "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s },  {z16.s, z17.s, z18.s, z19.s} \n\t"
                             "sqdmulh { z0.s, z1.s, z2.s, z3.s },  { z0.s, z1.s, z2.s, z3.s }, {z4.s, z5.s, z6.s, z7.s} \n\t"
                             "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s },  {z8.s, z9.s, z10.s, z11.s} \n\t"
                             "add     { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s },  z14.s \n\t"
                             "sqcvt   z0.h, { z0.s, z1.s} \n\t"
                             "sqcvt   z1.h, { z2.s, z3.s} \n\t"
                             "sclamp  {z0.h, z1.h}, z12.h, z13.h \n\t"
                             "st1h    {z0.h, z1.h}, pn9, [%[dst_ptr]] \n\t"
                             "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                             : [dst_ptr] "+r"(dst_ptr)
                             : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                             : "memory");

        za_index += 4;
        --i;
      }
    }
    else if (per_channel && !per_row)
    {
      const std::int32_t *mult_ptr  = &params.ruy_params->multiplier_fixedpoint[params.start_col];
      const std::int32_t *exp_ptr   = &params.ruy_params->multiplier_exponent[params.start_col];
      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        std::int32_t multiplier = *(mult_ptr++);
        std::int32_t exponent = *(exp_ptr++);
        __asm__ __volatile__(
            "mova     { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
            "dup      z10.s, %w[multiplier] \n\t"
            "dup      z11.s, %w[exponent] \n\t"
            "dup      z15.s, %w[minus_one] \n\t"
            "smin     z15.s, p7/m, z15.s, z11.s \n\t"
            "sub      z11.s, z11.s, z15.s \n\t"
            "srshl    { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z11.s \n\t"
            "sqdmulh  { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z10.s \n\t"
            "srshl    { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z15.s \n\t"
            "add      { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z14.s \n\t"
            "sqcvt    z0.h, { z0.s, z1.s} \n\t"
            "sqcvt    z1.h, { z2.s, z3.s} \n\t"
            "sclamp   {z0.h, z1.h}, z12.h, z13.h \n\t"
            "st1h     {z0.h, z1.h}, pn9, [%[dst_ptr]] \n\t"
            "add      %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
            : [dst_ptr] "+r"(dst_ptr)
            : [za_index] "r"(za_index), [multiplier] "r"(multiplier), [exponent] "r"(exponent),
              [dst_stride_bytes] "r"(dst_stride_bytes), [minus_one] "r"(minus_one)
            : "memory");

        za_index += 4;
        --i;
      }
    }
    else
    {
      std::int32_t multiplier = params.ruy_params->multiplier_fixedpoint[0];
      std::int32_t exp = params.ruy_params->multiplier_exponent[0];
      std::int32_t shr = ARM64_SME_MIN(-1, exp);
      std::int32_t shl = exp - shr;

      __asm__ __volatile__("dup  z10.s, %w[multiplier]  \n\t"
                           "dup  z11.s, %w[shl] \n\t"
                           "dup  z15.s, %w[shr] \n\t"
                           :
                           : [multiplier] "r"(multiplier), [shl] "r"(shl), [shr] "r"(shr)
                           : "memory");

      register uint32_t za_index asm("w12") = 0;
      while (i)
      {
        __asm__ __volatile__(
            "mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
            "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z11.s \n\t"
            "sqdmulh { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z10.s \n\t"
            "srshl   { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z15.s \n\t"
            "add     { z0.s, z1.s, z2.s, z3.s }, { z0.s, z1.s, z2.s, z3.s }, z14.s \n\t"
            "sqcvt   z0.h, { z0.s, z1.s} \n\t"
            "sqcvt   z1.h, { z2.s, z3.s} \n\t"
            "sclamp  {z0.h, z1.h}, z12.h, z13.h \n\t"
            "st1h    {z0.h, z1.h}, pn9, [%[dst_ptr]] \n\t"
            "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
            : [dst_ptr] "+r"(dst_ptr)
            : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
            : "memory");

        za_index += 4;
        --i;
      }
    }
  }

  FORCE_INLINE void sme_store_dst_singles_i32(const SME_Kernel_ParamsI8 &params)
  {
    gemmi8_idx_t N_elt_bytes      = params.num_rows << 2;
    gemmi8_idx_t N_vecs           = params.num_cols;
    void *dst_ptr                 = params.dst_ptr;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride << 2;
    uint32_t tile                 = 0;
    gemmi8_idx_t i                = N_vecs;
    uint64_t za_rows              = svcntb();

    register uint32_t za_index asm("w12") = 0;

    __asm__ __volatile__("whilelt p4.b, xzr, %[N_elt_bytes]" : : [N_elt_bytes] "r"(N_elt_bytes) : "memory", "cc");

    while (i >= 2)
    {
      __asm__ __volatile__("st1b    za0h.b[%w[za_index], 0], p4, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           "st1b    za0h.b[%w[za_index], 4], p4, [%[dst_ptr]] \n\t"
                           "add     %[dst_ptr], %[dst_ptr], %[dst_stride_bytes] \n\t"
                           : [dst_ptr] "+r"(dst_ptr)
                           : [za_index] "r"(za_index), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
      za_index += 8;
      if (za_index >= za_rows)
      {
        za_index = ++tile;
      }
      i -= 2;
    }
    if (i)
    {
      __asm__ __volatile__("st1b    za0h.b[%w[za_index], 0], p4, [%[dst_ptr]]"
                           :
                           : [za_index] "r"(za_index), [dst_ptr] "r"(dst_ptr), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
    }
  }

  FORCE_INLINE void sme_store_dst_pairs_i32(const SME_Kernel_ParamsI8 &params)
  {
    gemmi8_idx_t N_elts           = params.num_rows;
    gemmi8_idx_t N_vecs           = params.num_cols;
    void *dst_ptr                 = params.dst_ptr;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride << 2;
    gemmi8_idx_t i                = N_vecs;
    uint64_t za_rows              = svcntb();
    
    register uint32_t za_index asm("w12") = 0;

    __asm__ __volatile__("whilelt pn8.s, xzr, %[N_elts], vlx2" : : [N_elts] "r"(N_elts) : "memory", "cc");
    
    while (i >= 2)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b }, za0h.b[%w[za_index], 0:1] \n\t"
                           "mova    { z2.b, z3.b }, za0h.b[%w[za_index], 4:5] \n\t"
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
                           "st1w    { z0.s, z1.s }, pn8, [%[dst_ptr]] \n\t"
                           :
                           : [za_index] "r"(za_index), [dst_ptr] "r"(dst_ptr), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
    }
  }


  FORCE_INLINE void sme_store_dst_quads_i32(const SME_Kernel_ParamsI8 &params)
  {
    gemmi8_idx_t N_elts           = params.num_rows;
    gemmi8_idx_t N_vecs           = params.num_cols;
    void *dst_ptr                 = params.dst_ptr;
    gemmi8_idx_t dst_stride_bytes = params.ruy_params->dst_stride << 2;
    gemmi8_idx_t i                = N_vecs;
    
    register uint32_t za_index asm("w12") = 0;

    __asm__ __volatile__("whilelt pn8.s, xzr, %[N_elts], vlx4" : : [N_elts] "r"(N_elts) : "memory", "cc");
    
    while (i >= 2)
    {
      __asm__ __volatile__("mova    { z0.b, z1.b, z2.b, z3.b }, za0h.b[%w[za_index], 0:3] \n\t"
                           "mova    { z4.b, z5.b, z6.b, z7.b }, za0h.b[%w[za_index], 4:7] \n\t"
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
                           "st1w    { z0.s, z1.s, z2.s, z3.s }, pn8, [%[dst_ptr]] \n\t"
                           :
                           : [za_index] "r"(za_index), [dst_ptr] "r"(dst_ptr), [dst_stride_bytes] "r"(dst_stride_bytes)
                           : "memory");
    }
  }



  static void sme_gemmi8_NT_2x2_batch(const SME_Kernel_ParamsI8 &params)
  {
    const int8_t *lhs_ptr = (int8_t *)params.lhs_ptr;
    const int8_t *rhs_ptr = (int8_t *)params.rhs_ptr;

    // intro set predicates, zero ZA
    __asm__ __volatile__("zero    {za} \n\t"
                         "ptrue   p0.b \n\t"
                         : : : "memory", "cc");
    gemmi8_idx_t M4 = params.num_rows << 2;
    __asm__ __volatile__("whilelt pn9.b, xzr, %[M4], vlx2 " : : [M4] "r"(M4) : "memory", "cc");
    gemmi8_idx_t N4 = params.num_cols << 2;
    __asm__ __volatile__("whilelt pn10.b, xzr, %[N4], vlx2 " : : [N4] "r"(N4) : "memory", "cc");

    gemmi8_idx_t depth = params.ruy_params->depth;
    const gemmi8_idx_t threshold = 1;
    for (; depth >= threshold; depth -= 4)
    {
      __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn9/z, [%[lhs_ptr]]" : : [lhs_ptr] "r"(lhs_ptr) : "memory");
      lhs_ptr += params.ruy_params->lhs_stride;
      __asm__ __volatile__("ld1b    { z4.b, z5.b }, pn10/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");
      rhs_ptr += params.ruy_params->rhs_stride;

      __asm__ __volatile__("smopa   za0.s, p0/m, p0/m, z4.b, z0.b \n\t"
                           "smopa   za1.s, p0/m, p0/m, z4.b, z1.b \n\t"
                           "smopa   za2.s, p0/m, p0/m, z5.b, z0.b \n\t"
                           "smopa   za3.s, p0/m, p0/m, z5.b, z1.b \n\t"
                           : : : "memory");
    }

    __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn9/z, [%[lhs_ptr]]" : : [lhs_ptr] "r"(lhs_ptr) : "memory");
    __asm__ __volatile__("ld1b    { z4.b, z5.b }, pn10/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");

    __asm__ __volatile__("smopa   za0.s, p1/m, p1/m, z4.b, z0.b \n\t"
                         "smopa   za1.s, p1/m, p1/m, z4.b, z1.b \n\t"
                         "smopa   za2.s, p1/m, p1/m, z5.b, z0.b \n\t"
                         "smopa   za3.s, p1/m, p1/m, z5.b, z1.b \n\t"
                         : : : "memory");

    __asm__ __volatile__("whilelt pn8.s, xzr, %[num_rows], vlx2 \n\t"
                         "whilelt pn11.s, xzr, %[num_cols], vlx2 \n\t"
                         : : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_cols) : "memory", "cc");

    // Adding the bias
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      const int32_t *bias_ptr = NULL;
      if (!(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "addha za0.s, p0/m, p0/m, z0.s \n\t"
                             "addha za1.s, p0/m, p0/m, z1.s \n\t"
                             "addha za2.s, p0/m, p0/m, z0.s \n\t"
                             "addha za3.s, p0/m, p0/m, z1.s \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr)
                             : "memory");
      }
      else
      {
        bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z4.s, z5.s }, pn11/z, [%[bias_ptr]] \n\t"
                             "addva za0.s, p0/m, p0/m, z4.s \n\t"
                             "addva za1.s, p0/m, p0/m, z4.s \n\t"
                             "addva za2.s, p0/m, p0/m, z5.s \n\t"
                             "addva za3.s, p0/m, p0/m, z5.s \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr)
                             : "memory");
      }
    }

    // Subtract the rhs sums.
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_RHS_SUMS)
    {
      if (params.ruy_params->lhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->lhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->rhs_sums + params.start_col;
        __asm__ __volatile__("ld1w    { z8.s, z9.s }, pn11/z, [%[sums_ptr]] \n\t"
                             "dup  z6.s, %w[minus_zero_point] \n\t"
                             "dup  z4.s, %w[prod_zp_depth] \n\t"
                             "mov  z5.d, z4.d \n\t"
                             "mla  z4.s, p0/m ,z8.s, z6.s \n\t"
                             "mla  z5.s, p0/m ,z9.s, z6.s \n\t"
                             "addva za0.s, p0/m, p0/m, z4.s \n\t"
                             "addva za1.s, p0/m, p0/m, z4.s \n\t"
                             "addva za2.s, p0/m, p0/m, z5.s \n\t"
                             "addva za3.s, p0/m, p0/m, z5.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point), [prod_zp_depth] "r"(params.ruy_params->prod_zp_depth)
                             : "memory");
      }
    }

    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_LHS_SUMS)
    {
      if (params.ruy_params->rhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->rhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->lhs_sums + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[sums_ptr]] \n\t"
                             "dup  z4.s, %w[minus_zero_point] \n\t"
                             "mul  z0.s, z0.s, z4.s \n\t"
                             "mul  z1.s, z1.s, z4.s \n\t"
                             "addha za0.s, p0/m, p0/m, z0.s \n\t"
                             "addha za1.s, p0/m, p0/m, z1.s \n\t"
                             "addha za2.s, p0/m, p0/m, z0.s \n\t"
                             "addha za3.s, p0/m, p0/m, z1.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point)
                             : "memory");
      }
    }

    // store ZA data into C
    if (RUY_ASM_TYPE_ID_INT32 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_pairs_i32(params);
    }
    else if (RUY_ASM_TYPE_ID_INT16 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_pairs_i16(params);
    }

    if (RUY_ASM_TYPE_ID_INT8 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_pairs_i8<true>(params);
    }
    return sme_store_dst_pairs_i8<false>(params);
  }

  static void sme_gemmi8_NT_4x1_batch(const SME_Kernel_ParamsI8 &params)
  {
    const int8_t *lhs_ptr = (int8_t *)params.lhs_ptr;
    const int8_t *rhs_ptr = (int8_t *)params.rhs_ptr;

    // intro set predicates, zero ZA
    __asm__ __volatile__("zero    {za} \n\t"
                         "ptrue   p0.b \n\t"
                         : : : "memory", "cc");
    gemmi8_idx_t M4 = params.num_rows << 2;
    __asm__ __volatile__("whilelt pn9.b, xzr, %[M4], vlx4 " : : [M4] "r"(M4) : "memory", "cc");
    gemmi8_idx_t N4 = params.num_cols << 2;
    __asm__ __volatile__("whilelt p6.b, xzr, %[N4] " : : [N4] "r"(N4) : "memory", "cc");

    gemmi8_idx_t depth = params.ruy_params->depth;
    const gemmi8_idx_t threshold = 1;
    for (; depth >= threshold; depth -= 4)
    {
      __asm__ __volatile__("ld1b    { z0.b, z1.b, z2.b, z3.b }, pn9/z, [%[lhs_ptr]]" : : [lhs_ptr] "r"(lhs_ptr) : "memory");
      lhs_ptr += params.ruy_params->lhs_stride;
      __asm__ __volatile__("ld1b    {z4.b}, p6/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");
      rhs_ptr += params.ruy_params->rhs_stride;

      __asm__ __volatile__("smopa   za0.s, p0/m, p0/m, z4.b, z0.b \n\t"
                           "smopa   za1.s, p0/m, p0/m, z4.b, z1.b \n\t"
                           "smopa   za2.s, p0/m, p0/m, z4.b, z2.b \n\t"
                           "smopa   za3.s, p0/m, p0/m, z4.b, z3.b \n\t"
                           :
                           :
                           : "memory");
    }

    __asm__ __volatile__("whilelt pn8.s, xzr, %[num_rows], vlx4 \n\t"
                         "whilelt p7.s, xzr, %[num_cols] \n\t"
                         : : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_rows) : "memory", "cc");

    // Adding the bias
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      const int32_t *bias_ptr = NULL;
      if (!(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "addha za0.s, p0/m, p0/m, z0.s \n\t"
                             "addha za1.s, p0/m, p0/m, z1.s \n\t"
                             "addha za2.s, p0/m, p0/m, z2.s \n\t"
                             "addha za3.s, p0/m, p0/m, z3.s \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr)
                             : "memory");
      }
      else
      {
        bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z4.s }, p7/z, [%[bias_ptr]] \n\t"
                             "addva za0.s, p0/m, p0/m, z4.s \n\t"
                             "addva za1.s, p0/m, p0/m, z4.s \n\t"
                             "addva za2.s, p0/m, p0/m, z4.s \n\t"
                             "addva za3.s, p0/m, p0/m, z4.s \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr)
                             : "memory");
      }
    }
    // Subtract the rhs sums.
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_RHS_SUMS)
    {
      if (params.ruy_params->lhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->lhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->rhs_sums + params.start_col;
        __asm__ __volatile__("ld1w    { z7.s }, p0/z, [%[sums_ptr]] \n\t" // p0 ?
                             "dup  z6.s, %w[minus_zero_point] \n\t"
                             "dup  z4.s, %w[prod_zp_depth] \n\t"
                             "mla  z4.s, p0/m ,z7.s, z6.s \n\t"
                             "addva za0.s, p0/m, p0/m, z4.s \n\t"
                             "addva za1.s, p0/m, p0/m, z4.s \n\t"
                             "addva za2.s, p0/m, p0/m, z4.s \n\t"
                             "addva za3.s, p0/m, p0/m, z4.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point), [prod_zp_depth] "r"(params.ruy_params->prod_zp_depth)
                             : "memory");
      }
    }
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_LHS_SUMS)
    {
      if (params.ruy_params->rhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->rhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->lhs_sums + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[sums_ptr]] \n\t"
                             "dup  z4.s, %w[minus_zero_point] \n\t"
                             "mul  z0.s, z0.s, z4.s \n\t"
                             "mul  z1.s, z1.s, z4.s \n\t"
                             "mul  z2.s, z2.s, z4.s \n\t"
                             "mul  z3.s, z3.s, z4.s \n\t"
                             "addha za0.s, p0/m, p0/m, z0.s \n\t"
                             "addha za1.s, p0/m, p0/m, z1.s \n\t"
                             "addha za2.s, p0/m, p0/m, z2.s \n\t"
                             "addha za3.s, p0/m, p0/m, z3.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point)
                             : "memory");
      }
    }
    if (RUY_ASM_TYPE_ID_INT32 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_quads_i32(params);
    }
    else if (RUY_ASM_TYPE_ID_INT16 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_quads_i16(params);
    }

    if (RUY_ASM_TYPE_ID_INT8 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_quads_i8<true>(params);
    }
    return sme_store_dst_quads_i8<false>(params);
  }

  static void sme_gemmi8_NT_2x1_batch(const SME_Kernel_ParamsI8 &params)
  {
    const int8_t *lhs_ptr = (int8_t *)params.lhs_ptr;
    const int8_t *rhs_ptr = (int8_t *)params.rhs_ptr;
    const gemmi8_idx_t threshold = 1;
    gemmi8_idx_t depth = params.ruy_params->depth;

    // intro set predicates, zero ZA
    __asm__ __volatile__("zero    {za} \n\t"
                         "ptrue   p0.b \n\t"
                         : : : "memory", "cc");

    gemmi8_idx_t M4 = params.num_rows << 2;
    __asm__ __volatile__("whilelt pn9.b, xzr, %[M4], vlx2 " : : [M4] "r"(M4) : "memory", "cc");
    gemmi8_idx_t N4 = params.num_cols << 2;
    __asm__ __volatile__("whilelt p6.b, xzr, %[N4] " : : [N4] "r"(N4) : "memory", "cc");

    for (; depth >= threshold; depth -= 4)
    {
      __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn9/z, [%[lhs_ptr]]" : : [lhs_ptr] "r"(lhs_ptr) : "memory");
      lhs_ptr += params.ruy_params->lhs_stride;
      __asm__ __volatile__("ld1b    {z4.b}, p6/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");
      rhs_ptr += params.ruy_params->rhs_stride;

      __asm__ __volatile__("smopa   za0.s, p0/m, p0/m, z4.b, z0.b \n\t"
                           "smopa   za1.s, p0/m, p0/m, z4.b, z1.b \n\t"
                           :
                           :
                           : "memory");
    }

    // Adding the bias
    __asm__ __volatile__("whilelt pn8.s, xzr, %[num_rows], vlx2 \n\t"
                         "whilelt p7.s, xzr, %[num_cols] \n\t"
                         : : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_cols) : "memory", "cc");

    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      if (!(params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        const int32_t *bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "addha za0.s, p0/m, p0/m, z0.s \n\t"
                             "addha za1.s, p0/m, p0/m, z1.s \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr)
                             : "memory");
      }
      else
      {
        const int32_t *bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z4.s }, p7/z, [%[bias_ptr]] \n\t"
                             "addva za0.s, p0/m, p0/m, z4.s \n\t"
                             "addva za1.s, p0/m, p0/m, z4.s \n\t"
                             :
                             : [bias_ptr] "r"(bias_ptr)
                             : "memory");
      }
    }

    // Subtract the rhs sums.
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_RHS_SUMS)
    {
      if (params.ruy_params->lhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->lhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->rhs_sums + params.start_col;
        __asm__ __volatile__("ld1w    { z7.s }, p0/z, [%[sums_ptr]] \n\t" // p0 ?
                             "dup  z6.s, %w[minus_zero_point] \n\t"
                             "dup  z4.s, %w[prod_zp_depth] \n\t"
                             "mla  z4.s, p0/m ,z7.s, z6.s \n\t"
                             "addva za0.s, p0/m, p0/m, z4.s \n\t"
                             "addva za1.s, p0/m, p0/m, z4.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point), [prod_zp_depth] "r"(params.ruy_params->prod_zp_depth)
                             : "memory");
      }
    }

    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_LHS_SUMS)
    {
      if (params.ruy_params->rhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->rhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->lhs_sums + params.start_row;
        __asm__ __volatile__("ld1w    { z0.s, z1.s }, pn8/z, [%[sums_ptr]] \n\t"
                             "dup  z4.s, %w[minus_zero_point] \n\t"
                             "mul  z0.s, z0.s, z4.s \n\t"
                             "mul  z1.s, z1.s, z4.s \n\t"
                             "addha za0.s, p0/m, p0/m, z0.s \n\t"
                             "addha za1.s, p0/m, p0/m, z1.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point)
                             : "memory");
      }
    }

    // store ZA data into C
    if (RUY_ASM_TYPE_ID_INT32 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_pairs_i32(params);
    }
    else if (RUY_ASM_TYPE_ID_INT16 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_pairs_i16(params);
    }

    if (RUY_ASM_TYPE_ID_INT8 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_pairs_i8<true>(params);
    }
    return sme_store_dst_pairs_i8<false>(params);
  }

  template <unsigned NUM_TILES>
  static void sme_gemmi8_NT_1xN_batch(const SME_Kernel_ParamsI8 &params)
  {
    static_assert(NUM_TILES >= 1 && NUM_TILES <= 4, "Expect NUM_TILES 1-4");

    const int8_t *lhs_ptr = (int8_t *)params.lhs_ptr;
    const int8_t *rhs_ptr = (int8_t *)params.rhs_ptr;
    const gemmi8_idx_t threshold = 1;
    gemmi8_idx_t depth = params.ruy_params->depth;

    // intro set predicates, zero ZA
    gemmi8_idx_t M4 = params.num_rows << 2;
    __asm__ __volatile__("zero    {za} \n\t"
                         "ptrue   p0.b \n\t"
                         "whilelt p4.b, xzr, %[M4] \n\t"
                         :
                         : [M4] "r"(M4)
                         : "memory", "cc");
    __asm__ __volatile__("whilelt p5.b, xzr, %[M4] " : : [M4] "r"(M4) : "memory", "cc");

    gemmi8_idx_t N4 = params.num_cols << 2;
    if (NUM_TILES == 1)
    {
      __asm__ __volatile__("whilelt p7.b, xzr, %[N4] " : : [N4] "r"(N4) : "memory", "cc");
    }
    else if (NUM_TILES == 2)
    {
      __asm__ __volatile__("whilelt pn9.b, xzr, %[N4], vlx2 " : : [N4] "r"(N4) : "memory", "cc");
    }
    else
    {
      __asm__ __volatile__("whilelt pn9.b, xzr, %[N4], vlx4 " : : [N4] "r"(N4) : "memory", "cc");
    }

    for (; depth >= threshold; depth -= 4)
    {
      if (NUM_TILES == 1)
      {
        __asm__ __volatile__("ld1b    { z0.b }, p7/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");
      }
      else if (NUM_TILES == 2)
      {
        __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn9/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");
      }
      else
      {
        __asm__ __volatile__("ld1b    { z0.b, z1.b, z2.b, z3.b }, pn9/z, [%[rhs_ptr]]" : : [rhs_ptr] "r"(rhs_ptr) : "memory");
      }
      rhs_ptr += params.ruy_params->rhs_stride;

      __asm__ __volatile__("ld1b    {z4.b}, p5/z, [%[lhs_ptr]]" : : [lhs_ptr] "r"(lhs_ptr) : "memory");
      lhs_ptr += params.ruy_params->lhs_stride;

      if (NUM_TILES == 1)
      {
        __asm__ __volatile__("smopa   za0.s, p0/m, p4/m, z0.b, z4.b" : : : "memory");
      }
      else if (NUM_TILES == 2)
      {
        __asm__ __volatile__("smopa   za0.s, p0/m, p4/m, z0.b, z4.b \n\t"
                             "smopa   za1.s, p0/m, p4/m, z1.b, z4.b \n\t"
                             : : : "memory");
      }
      else if (NUM_TILES == 3)
      {
        __asm__ __volatile__("smopa   za0.s, p0/m, p4/m, z0.b, z4.b \n\t"
                             "smopa   za1.s, p0/m, p4/m, z1.b, z4.b \n\t"
                             "smopa   za2.s, p0/m, p4/m, z2.b, z4.b \n\t"
                             : : : "memory");
      }
      else
      {
        __asm__ __volatile__("smopa   za0.s, p0/m, p4/m, z0.b, z4.b \n\t"
                             "smopa   za1.s, p0/m, p4/m, z1.b, z4.b \n\t"
                             "smopa   za2.s, p0/m, p4/m, z2.b, z4.b \n\t"
                             "smopa   za3.s, p0/m, p4/m, z3.b, z4.b \n\t"
                             : : : "memory");
      }
    }

    // Adding the bias
    __asm__ __volatile__("whilelt pn8.s, xzr, %[num_cols], vlx4 \n\t"
                         "whilelt p7.s, xzr, %[num_rows] \n\t"
                         : : [num_rows] "r"(params.num_rows), [num_cols] "r"(params.num_cols) : "memory", "cc");

    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_BIAS)
    {
      if ((params.ruy_params->flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL))
      {
        const int32_t *bias_ptr = params.ruy_params->bias + params.start_col;
        __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[bias_ptr]] \n\t"
                             "addva za0.s, p0/m, p0/m, z0.s \n\t"
                             "addva za1.s, p0/m, p0/m, z1.s \n\t"
                             "addva za2.s, p0/m, p0/m, z2.s \n\t"
                             "addva za3.s, p0/m, p0/m, z3.s \n\t"
                             : : [bias_ptr] "r"(bias_ptr) : "memory");
      }
      else
      {
        const int32_t *bias_ptr = params.ruy_params->bias + params.start_row;
        __asm__ __volatile__("ld1w    { z4.s }, p7/z, [%[bias_ptr]] \n\t"
                             "addha za0.s, p0/m, p0/m, z4.s \n\t"
                             "addha za1.s, p0/m, p0/m, z4.s \n\t"
                             "addha za2.s, p0/m, p0/m, z4.s \n\t"
                             "addha za3.s, p0/m, p0/m, z4.s \n\t"
                             : : [bias_ptr] "r"(bias_ptr) : "memory");
      }
    }
    // Subtract the rhs sums.
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_RHS_SUMS)
    {
      if (params.ruy_params->lhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->lhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->rhs_sums + params.start_col;
        __asm__ __volatile__("ld1w    { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[sums_ptr]] \n\t"
                             "dup  z4.s, %w[minus_zero_point] \n\t"
                             "mul  z0.s, z0.s, z4.s \n\t"
                             "mul  z1.s, z1.s, z4.s \n\t"
                             "mul  z2.s, z2.s, z4.s \n\t"
                             "mul  z3.s, z3.s, z4.s \n\t"
                             "addva za0.s, p0/m, p0/m, z0.s \n\t"
                             "addva za1.s, p0/m, p0/m, z1.s \n\t"
                             "addva za2.s, p0/m, p0/m, z2.s \n\t"
                             "addva za3.s, p0/m, p0/m, z3.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point)
                             : "memory");
      }
    }
    if (params.ruy_params->flags & RUY_ASM_FLAG_HAS_LHS_SUMS)
    {
      if (params.ruy_params->rhs_zero_point)
      {
        int32_t minus_zero_point = -params.ruy_params->rhs_zero_point;
        const int32_t *sums_ptr = params.ruy_params->lhs_sums + params.start_row;
        __asm__ __volatile__("ld1w    { z7.s }, p0/z, [%[sums_ptr]] \n\t" // p0 ?
                             "dup  z6.s, %w[minus_zero_point] \n\t"
                             "dup  z4.s, %w[prod_zp_depth] \n\t"
                             "mla  z4.s, p0/m ,z7.s, z6.s \n\t"
                             "addha za0.s, p0/m, p0/m, z4.s \n\t"
                             "addha za1.s, p0/m, p0/m, z4.s \n\t"
                             "addha za2.s, p0/m, p0/m, z4.s \n\t"
                             "addha za3.s, p0/m, p0/m, z4.s \n\t"
                             :
                             : [sums_ptr] "r"(sums_ptr), [minus_zero_point] "r"(minus_zero_point), [prod_zp_depth] "r"(params.ruy_params->prod_zp_depth)
                             : "memory");
      }
    }

    // store ZA data into C
    if (RUY_ASM_TYPE_ID_INT32 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_singles_i32(params);
    }
    else  if (RUY_ASM_TYPE_ID_INT16 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_singles_i16(params);
    }

    if (RUY_ASM_TYPE_ID_INT8 == params.ruy_params->dst_type_id)
    {
      return sme_store_dst_singles_i8<true>(params);
    }
    return sme_store_dst_singles_i8<false>(params);
  }

  // The main function of the SME GEMMi8. 
  static void gemmi8_sme(const SME_RUY_Kernel_Params8Bits &ruy_params, gemmi8_idx_t num_rows, gemmi8_idx_t num_cols)
  {
    SME_Kernel_ParamsI8 params;

    int dst_elem_num_bytes = 0;
    if (!getOutputNumBytes(ruy_params, dst_elem_num_bytes))
    {
      return;
    }

    params.ruy_params = &ruy_params;
    const auto orig_start_row = ruy_params.start_row;
    const auto orig_start_col = ruy_params.start_col;

    const gemmi8_idx_t elements_count1 = svcntw();
    const gemmi8_idx_t elements_count2 = elements_count1 * 2;
    const gemmi8_idx_t elements_count3 = elements_count1 * 3;
    const gemmi8_idx_t elements_count4 = elements_count1 * 4;
    gemmi8_idx_t i = 0;

    // In case number of rows is big enough we use sme_sgemm_NT_4x1_batch.
    for (; num_rows > i + elements_count3; i += elements_count4)
    {
      gemmi8_idx_t j = 0;
      const int8_t *lhs_ptr = params.ruy_params->lhs_base_ptr + i * 4;
      for (; j < num_cols; j += elements_count1)
      {
        params.lhs_ptr = lhs_ptr;
        params.num_rows = ARM64_SME_MIN(num_rows - i, elements_count4);
        params.num_cols = ARM64_SME_MIN(num_cols - j, elements_count1);
        params.rhs_ptr = (int8_t *)params.ruy_params->rhs_base_ptr + j * 4;
        params.dst_ptr = (int8_t *)params.ruy_params->dst_base_ptr + dst_idx(i, j, params.ruy_params->dst_stride, dst_elem_num_bytes);
        params.start_row = orig_start_row + i;
        params.start_col = orig_start_col + j;
        sme_gemmi8_NT_4x1_batch(params);
      }
    }

    // Next we try to use sme_sgemm_NT_2x2_batch.
    for (; num_rows > i + elements_count1; i += elements_count2)
    {
      params.lhs_ptr = params.ruy_params->lhs_base_ptr + i * 4;
      params.num_rows = ARM64_SME_MIN(num_rows - i, elements_count2);
      params.start_row = orig_start_row + i;
      gemmi8_idx_t j = 0;
      for (; num_cols > j + elements_count1; j += elements_count2)
      {
        params.num_cols = ARM64_SME_MIN(num_cols - j, elements_count2);
        params.rhs_ptr = (int8_t *)params.ruy_params->rhs_base_ptr + j * 4;
        params.dst_ptr = (int8_t *)params.ruy_params->dst_base_ptr + dst_idx(i, j, params.ruy_params->dst_stride, dst_elem_num_bytes);
        params.start_col = orig_start_col + j;
        sme_gemmi8_NT_2x2_batch(params);
      }

      // For last cols we use sme_sgemm_NT_2x1_batch.
      if (j < num_cols)
      {
        params.num_cols = ARM64_SME_MIN(num_cols - j, elements_count1);
        params.rhs_ptr = (int8_t *)params.ruy_params->rhs_base_ptr + j * 4;
        params.dst_ptr = (int8_t *)params.ruy_params->dst_base_ptr + dst_idx(i, j, params.ruy_params->dst_stride, dst_elem_num_bytes);
        params.start_col = orig_start_col + j;
        sme_gemmi8_NT_2x1_batch(params);
      }
    }

    // The "leftover" case - handling last rows
    for (; i < num_rows; i += elements_count1)
    {
      params.lhs_ptr = params.ruy_params->lhs_base_ptr + i * 4;
      params.start_row = orig_start_row + i;
      gemmi8_idx_t j = 0;
      for (; num_cols > j + elements_count3; j += elements_count4)
      {
        params.num_rows = ARM64_SME_MIN(num_rows - i, elements_count1);
        params.num_cols = ARM64_SME_MIN(num_cols - j, elements_count4);
        params.rhs_ptr = (int8_t *)params.ruy_params->rhs_base_ptr + j * 4;
        params.dst_ptr = (int8_t *)params.ruy_params->dst_base_ptr + dst_idx(i, j, params.ruy_params->dst_stride, dst_elem_num_bytes);
        params.start_col = orig_start_col + j;
        sme_gemmi8_NT_1xN_batch<4>(params);
      }
      if (j < num_cols)
      {
        params.num_rows = ARM64_SME_MIN(num_rows - i, elements_count1);
        params.num_cols = ARM64_SME_MIN(num_cols - j, elements_count4);
        params.rhs_ptr = (int8_t *)params.ruy_params->rhs_base_ptr + j * 4;
        params.dst_ptr = (int8_t *)params.ruy_params->dst_base_ptr + dst_idx(i, j, params.ruy_params->dst_stride, dst_elem_num_bytes);
        params.start_col = orig_start_col + j;
        if (num_cols - j > elements_count2)
        {
          sme_gemmi8_NT_1xN_batch<3>(params);
        }
        else if (num_cols - j > elements_count1)
        {
          sme_gemmi8_NT_1xN_batch<2>(params);
        }
        else
        {
          sme_gemmi8_NT_1xN_batch<1>(params);
        }
      }
    }
  }

  // The entry function to the GEMMi8 SME kernel.
  void Kernel8bitArm64SME(const SME_RUY_Kernel_Params8Bits &params, int num_rows, int num_cols)
  {
    profiler::ScopeLabel label("Kernel (kArm64Sme)");
    uint8_t abi_stack[64] __attribute__((aligned(16)));
    SMSTART(abi_stack);
    gemmi8_sme(params, num_rows, num_cols);
    SMSTOP(abi_stack);

    return;
  }

} // namespace ruy

#endif // RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)
