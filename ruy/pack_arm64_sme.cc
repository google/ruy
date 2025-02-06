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
#include <cstring>

#include "ruy/opt_set.h"
#include "ruy/pack_common.h"
#include "ruy/pack_arm.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/profiler/instrumentation.h"


// SME packing applies both data reordering and cols/rows accumulation.
// The SME fixed point kernels use FixedKernelLayout with kRows = 4.
// This enables us reduce the number of operations to read the values for GEMMN.
// Notice we use gemvi8_packed_sme to calculate the rows/cols sums for GEMMi8 - 
// this is a GEMV kernel we use with a vector of 1s.

#if RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)
#include "ruy/arm64_sme_common.h"
namespace ruy
{
  template <int num_trailing_quads, bool trailing_pair>
  static void sme_addi8_packed_batch_impl(gemvi8_packed_idx_t num_cols, const int32_t *src_ptr_base,
                                          gemvi8_packed_idx_t src_stride, gemvi8_packed_idx_t num_rows, gemvi8_packed_idx_t quad4)
  {
    const gemvi8_packed_idx_t elements_count = svcntw();
    const gemvi8_packed_idx_t elements_count4 = elements_count * 4;
    const gemvi8_packed_idx_t elements_count8 = elements_count * 8;
    const gemvi8_packed_idx_t elements_count12 = elements_count * 12;
    const gemvi8_packed_idx_t elements_count16 = elements_count * 16;

    __asm__ __volatile__("ptrue pn8.s \n\t"
                         "dup z8.b, #1 \n\t"
                         : : : "memory");
    if (trailing_pair)
    {
      if (num_trailing_quads)
      {
        __asm__ __volatile__("ptrue pn9.s" : : : "memory", "cc");
      }
      gemvi8_packed_idx_t remaining = (quad4 * 4 + num_trailing_quads) * elements_count4;
      __asm__ __volatile__("whilelt pn10.s, %[remaining], %[num_rows], vlx2 " : : [remaining] "r"(remaining), [num_rows] "r"(num_rows) : "memory", "cc");
    }
    else if (num_trailing_quads)
    {
      gemvi8_packed_idx_t remaining = (quad4 * 4 + num_trailing_quads - 1) * elements_count4;
      __asm__ __volatile__("whilelt pn9.s, %[remaining], %[num_rows], vlx4 " : : [remaining] "r"(remaining), [num_rows] "r"(num_rows) : "memory", "cc");
    }

    for (gemvi8_packed_idx_t idx = 0; idx < num_cols; idx += 4)
    {
      const int32_t *src_ptr = src_ptr_base;
      register uint32_t za_index asm("w8");
      za_index = 0;
      for (gemvi8_packed_idx_t k = 0; k < quad4; ++k)
      {
        __asm__ __volatile__("ld1w  { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[src_ptr], #0, mul vl] \n\t"
                             "ld1w  { z4.s, z5.s, z6.s, z7.s }, pn8/z, [%[src_ptr], #4, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 0], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             "sdot  za.s[%w[za_index], 1], { z4.b, z5.b, z6.b, z7.b }, z8.b \n\t"
                             "ld1w  { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[src_ptr], #8, mul vl] \n\t"
                             "ld1w  { z4.s, z5.s, z6.s, z7.s }, pn8/z, [%[src_ptr], #12, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 2], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             "sdot  za.s[%w[za_index], 3], { z4.b, z5.b, z6.b, z7.b }, z8.b \n\t"
                             :
                             : [src_ptr] "r"(src_ptr), [za_index] "r"(za_index)
                             : "memory");
        src_ptr += elements_count16;
        za_index += 4;
      }
      if (num_trailing_quads == 4)
      {
        __asm__ __volatile__("ld1w  { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[src_ptr], #0, mul vl] \n\t"
                             "ld1w  { z4.s, z5.s, z6.s, z7.s }, pn8/z, [%[src_ptr], #4, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 0], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             "sdot  za.s[%w[za_index], 1], { z4.b, z5.b, z6.b, z7.b }, z8.b \n\t"
                             "ld1w  { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[src_ptr], #8, mul vl] \n\t"
                             "ld1w  { z4.s, z5.s, z6.s, z7.s }, pn9/z, [%[src_ptr], #12, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 2], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             "sdot  za.s[%w[za_index], 3], { z4.b, z5.b, z6.b, z7.b }, z8.b \n\t"
                             :
                             : [src_ptr] "r"(src_ptr), [za_index] "r"(za_index)
                             : "memory");
      }
      if (num_trailing_quads == 3)
      {
        __asm__ __volatile__("ld1w  { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[src_ptr], #0, mul vl] \n\t"
                             "ld1w  { z4.s, z5.s, z6.s, z7.s }, pn8/z, [%[src_ptr], #4, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 0], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             "sdot  za.s[%w[za_index], 1], { z4.b, z5.b, z6.b, z7.b }, z8.b \n\t"
                             "ld1w  { z0.s, z1.s, z2.s, z3.s }, pn9/z, [%[src_ptr], #8, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 2], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             :
                             : [src_ptr] "r"(src_ptr), [za_index] "r"(za_index)
                             : "memory");
        src_ptr += elements_count12;
        za_index += 3;
      }
      else if (num_trailing_quads == 2)
      {
        __asm__ __volatile__("ld1w  { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[src_ptr], #0, mul vl] \n\t"
                             "ld1w  { z4.s, z5.s, z6.s, z7.s }, pn9/z, [%[src_ptr], #4, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 0], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             "sdot  za.s[%w[za_index], 1], { z4.b, z5.b, z6.b, z7.b }, z8.b \n\t"
                             :
                             : [src_ptr] "r"(src_ptr), [za_index] "r"(za_index)
                             : "memory");
        src_ptr += elements_count8;
        za_index += 2;
      }
      else if (num_trailing_quads == 1)
      {
        __asm__ __volatile__("ld1w  { z0.s, z1.s, z2.s, z3.s }, pn9/z, [%[src_ptr], #0, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 0], { z0.b, z1.b, z2.b, z3.b }, z8.b \n\t"
                             :
                             : [src_ptr] "r"(src_ptr), [za_index] "r"(za_index)
                             : "memory");
        src_ptr += elements_count4;
        za_index += 1;
      }
      if (trailing_pair)
      {
        __asm__ __volatile__("ld1w  { z0.s, z1.s }, pn10/z, [%[src_ptr], #0, mul vl] \n\t"
                             "sdot  za.s[%w[za_index], 0], { z0.b, z1.b }, z8.b \n\t"
                             :
                             : [src_ptr] "r"(src_ptr), [za_index] "r"(za_index)
                             : "memory");
      }
      src_ptr_base += src_stride;
    }
  }

  static void sme_gemvi8_packed_batch(gemvi8_packed_idx_t N, const int32_t *src_ptr, gemvi8_packed_idx_t src_stride, 
                                      gemvi8_packed_idx_t num_rows, bool trailing_pair)
  {
    const gemvi8_packed_idx_t elements_count = svcntw();
    const gemvi8_packed_idx_t elements_count4 = elements_count * 4;
    const gemvi8_packed_idx_t elements_count16 = elements_count * 16;
    gemvi8_packed_idx_t trailing4q = num_rows % elements_count16;
    gemvi8_packed_idx_t quad4 = num_rows / elements_count16;
    typedef void (*FP)(gemvi8_packed_idx_t n, const int32_t *Aptr, gemvi8_packed_idx_t src_stride, gemvi8_packed_idx_t M, gemvi8_packed_idx_t quad4);
    FP batch_func;

    if (trailing_pair)
    {
      static const FP fundst_ptr[4] = {
          sme_addi8_packed_batch_impl<0, 1>,
          sme_addi8_packed_batch_impl<1, 1>,
          sme_addi8_packed_batch_impl<2, 1>,
          sme_addi8_packed_batch_impl<3, 1>};
      batch_func = fundst_ptr[trailing4q / elements_count4];
    }
    else
    {
      static const FP fundst_ptr[5] = {
          sme_addi8_packed_batch_impl<0, 0>,
          sme_addi8_packed_batch_impl<1, 0>,
          sme_addi8_packed_batch_impl<2, 0>,
          sme_addi8_packed_batch_impl<3, 0>,
          sme_addi8_packed_batch_impl<4, 0>};
      batch_func = fundst_ptr[(trailing4q + elements_count4 - 1) / elements_count4];
    }

    batch_func(N, src_ptr, src_stride, num_rows, quad4);
  }

  static void sme_store_gemvi8_packed_res(int32_t *Y_ptr, gemvi8_packed_idx_t n_full_quads, gemvi8_packed_idx_t remaining, bool trailing_pair)
  {
    bool trailing_quad = remaining != 0 && !trailing_pair;
    __asm__ __volatile__("ptrue pn8.s " : : : "memory");
    register uint32_t za_index asm("w8");
    za_index = 0;
    for (; za_index + 2 <= n_full_quads; za_index += 2)
    {
      __asm__ __volatile__("mova { z0.d, z1.d, z2.d, z3.d }, za.d[%w[za_index], 0] \n\t"
                           "mova { z4.d, z5.d, z6.d, z7.d }, za.d[%w[za_index], 1] \n\t"
                           "st1w { z0.s, z1.s, z2.s, z3.s }, pn8, [%[Y_ptr], #0, mul vl] \n\t"
                           "st1w { z4.s, z5.s, z6.s, z7.s }, pn8, [%[Y_ptr], #4, mul vl] \n\t"
                           "incb %[Y_ptr], all, mul #8 \n\t"
                           : [Y_ptr] "+r"(Y_ptr)
                           : [za_index] "r"(za_index)
                           : "memory");
    }
    if (za_index < n_full_quads)
    {
      __asm__ __volatile__("mova { z0.d, z1.d, z2.d, z3.d }, za.d[%w[za_index], 0] \n\t"
                           "st1w { z0.s, z1.s, z2.s, z3.s }, pn8, [%[Y_ptr], #0, mul vl] \n\t"
                           "incb %[Y_ptr], all, mul #4 \n\t"
                           : [Y_ptr] "+r"(Y_ptr)
                           : [za_index] "r"(za_index)
                           : "memory");
      za_index++;
    }
    if (trailing_quad)
    {
      __asm__ __volatile__("whilelt pn8.s, xzr, %[remaining], vlx4 \n\t"
                           "mova { z0.d, z1.d, z2.d, z3.d }, za.d[%w[za_index], 0] \n\t"
                           "st1w { z0.s, z1.s, z2.s, z3.s }, pn8, [%[Y_ptr], #0, mul vl] \n\t"
                           :
                           : [Y_ptr] "r"(Y_ptr), [za_index] "r"(za_index), [remaining] "r"(remaining)
                           : "memory", "cc");
    }
    if (trailing_pair)
    {
      __asm__ __volatile__("whilelt pn8.s, xzr, %[remaining], vlx2 \n\t"
                           "mova { z0.d, z1.d }, za.d[%w[za_index], 0] \n\t"
                           "st1w { z0.s, z1.s }, pn8, [%[Y_ptr], #0, mul vl] \n\t"
                           :
                           : [Y_ptr] "r"(Y_ptr), [za_index] "r"(za_index), [remaining] "r"(remaining)
                           : "memory", "cc");
    }
  }

  static void sme_load_gemvi8_packed_res(int32_t *Y_ptr, gemvi8_packed_idx_t n_full_quads, gemvi8_packed_idx_t remaining, bool trailing_pair)
  {
    bool trailing_quad = remaining != 0 && !trailing_pair;
    __asm__ __volatile__("ptrue pn8.s " : : : "memory");
    register uint32_t za_index asm("w8");
    za_index = 0;
    for (; za_index + 2 <= n_full_quads; za_index += 2)
    {
      __asm__ __volatile__("ld1w { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[Y_ptr], #0, mul vl] \n\t"
                           "ld1w { z4.s, z5.s, z6.s, z7.s }, pn8/z, [%[Y_ptr], #4, mul vl] \n\t"
                           "mova za.d[%w[za_index], 0], { z0.d, z1.d, z2.d, z3.d } \n\t"
                           "mova za.d[%w[za_index], 1], { z4.d, z5.d, z6.d, z7.d } \n\t"
                           "incb %[Y_ptr], all, mul #8 \n\t"
                           : [Y_ptr] "+r"(Y_ptr)
                           : [za_index] "r"(za_index)
                           : "memory");
    }
    if (za_index < n_full_quads)
    {
      __asm__ __volatile__("ld1w { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[Y_ptr], #0, mul vl] \n\t"
                           "mova za.d[%w[za_index], 0], { z0.d, z1.d, z2.d, z3.d } \n\t"
                           "incb %[Y_ptr], all, mul #4 \n\t"
                           : [Y_ptr] "+r"(Y_ptr)
                           : [za_index] "r"(za_index)
                           : "memory");
      za_index++;
    }
    if (trailing_quad)
    {
      __asm__ __volatile__("whilelt pn8.s, xzr, %[remaining], vlx4 \n\t"
                           "ld1w { z0.s, z1.s, z2.s, z3.s }, pn8/z, [%[Y_ptr], #0, mul vl] \n\t"
                           "mova za.d[%w[za_index], 0], { z0.d, z1.d, z2.d, z3.d } \n\t"
                           :
                           : [Y_ptr] "r"(Y_ptr), [za_index] "r"(za_index), [remaining] "r"(remaining)
                           : "memory", "cc");
    }
    if (trailing_pair)
    {
      __asm__ __volatile__("whilelt pn8.s, xzr, %[remaining], vlx2 \n\t"
                           "ld1w { z0.s, z1.s }, pn8/z, [%[Y_ptr], #0, mul vl] \n\t"
                           "mova za.d[%w[za_index], 0], { z0.d, z1.d } \n\t"
                           :
                           : [Y_ptr] "r"(Y_ptr), [za_index] "r"(za_index), [remaining] "r"(remaining)
                           : "memory", "cc");
    }
  }

  __attribute__((noinline)) void gemvi8_packed_sme_impl(gemvi8_packed_idx_t M, gemvi8_packed_idx_t N,
                                                        const int32_t *A, gemvi8_packed_idx_t src_stride,
                                                        int32_t *Y, bool init_with_Y)
  {
    // To maintain some locality do work in partial sums if matrix is very big.
    const gemvi8_packed_idx_t MAT_THRESHOLD = 1024 * 1024 * 8;
    const gemvi8_packed_idx_t MAX_COLS_PATRIAL_SUM = 1024;
    const gemvi8_packed_idx_t elements_count = svcntw();
    const gemvi8_packed_idx_t MAX_ROWS = elements_count * elements_count * 4;
    
    // Need 4 quads to run full throughput (4 cycle acc latency)
    const gemvi8_packed_idx_t MIN_ROWS = 4 * 4 * elements_count;

    gemvi8_packed_idx_t matrix_sz = N * src_stride * sizeof(int8_t);
    gemvi8_packed_idx_t max_cols = N;
    if (M > MAX_ROWS && matrix_sz > MAT_THRESHOLD)
    {
      max_cols = MAX_COLS_PATRIAL_SUM;
    }

    for (gemvi8_packed_idx_t first_col = 0; first_col < N; first_col += max_cols)
    {
      const int32_t *A0 = A + ((first_col * src_stride) >> 2);
      gemvi8_packed_idx_t n0 = first_col + max_cols > N ? (N - first_col) : max_cols;
      gemvi8_packed_idx_t row = 0;
      while (row < M)
      {
        // calc how many quad ops to process in this batch.
        gemvi8_packed_idx_t num_rows = M - row;
        if (num_rows > MAX_ROWS)
        {
          if (num_rows < MAX_ROWS + MIN_ROWS)
          {
            num_rows = MAX_ROWS >> 1;
          }
          else
          {
            num_rows = MAX_ROWS;
          }
        }

        const gemvi8_packed_idx_t elements_count = svcntw();
        const gemvi8_packed_idx_t elements_count2 = elements_count * 2;
        const gemvi8_packed_idx_t elements_count4 = elements_count * 4;
        gemvi8_packed_idx_t trailing1q = num_rows % elements_count4;
        gemvi8_packed_idx_t n_full_quads = num_rows / elements_count4;
        bool trailing_pair = trailing1q > 0 && trailing1q <= elements_count2;
        
        // If needed  initialize the array with prior partial sums. else intialize ZA with 0.
        if (first_col > 0 || init_with_Y)
        {
          sme_load_gemvi8_packed_res(&Y[row], n_full_quads, trailing1q, trailing_pair);
        }
        else
        {
          asm volatile("zero    {za}" : : : "memory");
        }

        // Compute the current batch of results
        sme_gemvi8_packed_batch(n0, &A0[row], src_stride, num_rows, trailing_pair);

        // Store results into y.
        sme_store_gemvi8_packed_res(&Y[row], n_full_quads, trailing1q, trailing_pair);

        row += num_rows;
      }
    }
  }

  void gemvi8_packed_sme(gemvi8_packed_idx_t M, gemvi8_packed_idx_t N,
                         const int32_t *A, gemvi8_packed_idx_t src_stride,
                         int32_t *Y, bool init_with_Y)
  {
    uint8_t abi_stack[64] __attribute__((aligned(16)));
    SMSTART(abi_stack);
    gemvi8_packed_sme_impl(M, N, A, src_stride, Y, init_with_Y);
    SMSTOP(abi_stack);
  }

  static void sme_tranpose_32_st(void *T_ptr, transpose_idx_t LDT_bytes, uint32_t st_tile, transpose_idx_t len, transpose_idx_t count)
  {
#define ST4_TILE_IMPL(tile, limit)                                                                    \
  za_index = 0;                                                                                       \
  for (; za_index < limit; za_index += 4)                                                             \
  {                                                                                                   \
    __asm__ __volatile__("mova    { z0.s, z1.s, z2.s, z3.s }, za" #tile "v.s[%w[za_index], 0:3] \n\t" \
                         "st1w    { z0.s }, p0, [%[T_ptr]]; \n\t"                                     \
                         "add     %[T_ptr], %[T_ptr], %[LDT_bytes] \n\t"                              \
                         "st1w    { z1.s }, p0, [%[T_ptr]]; \n\t"                                     \
                         "add     %[T_ptr], %[T_ptr], %[LDT_bytes] \n\t"                              \
                         "st1w    { z2.s }, p0, [%[T_ptr]]; \n\t"                                     \
                         "add     %[T_ptr], %[T_ptr], %[LDT_bytes] \n\t"                              \
                         "st1w    { z3.s }, p0, [%[T_ptr]]; \n\t"                                     \
                         "add     %[T_ptr], %[T_ptr], %[LDT_bytes] \n\t"                              \
                         : [za_index] "+r"(za_index), [T_ptr] "+r"(T_ptr)                             \
                         : [LDT_bytes] "r"(LDT_bytes)                                                 \
                         : "memory");                                                                 \
  }

#define ST_TILE_REM_IMPL(tile, limit)                                                    \
  for (; za_index < limit; ++za_index)                                                   \
  {                                                                                      \
    __asm__ __volatile__("st1w    za" #tile "v.s[%w[za_index], 0], p0, [%[T_ptr]]; \n\t" \
                         "add     %[T_ptr], %[T_ptr], %[LDT_bytes] \n\t"                 \
                         : [za_index] "+r"(za_index), [T_ptr] "+r"(T_ptr)                \
                         : [LDT_bytes] "r"(LDT_bytes)                                    \
                         : "memory");                                                    \
  }

    __asm__ __volatile__("whilelt p0.s, xzr, %[len] \n\t"
                         :
                         : [len] "r"(len)
                         : "memory", "cc");
    register uint32_t za_index asm("w12");
    transpose_idx_t ELT_CNT1 = svcntw();
    if (count > ELT_CNT1)
    {
      uint32_t limit = ARM64_SME_MIN(count - ELT_CNT1, ELT_CNT1);
      uint32_t limit4 = limit - (limit % 4);
      if (st_tile)
      {
        ST4_TILE_IMPL(2, ELT_CNT1)
        ST4_TILE_IMPL(3, limit4)
        ST_TILE_REM_IMPL(3, limit)
      }
      else
      {
        ST4_TILE_IMPL(0, ELT_CNT1)
        ST4_TILE_IMPL(1, limit4)
        ST_TILE_REM_IMPL(1, limit);
      }
    }
    else
    {
      uint32_t limit = (uint32_t)ARM64_SME_MIN(count, ELT_CNT1);
      uint32_t limit4 = limit - (limit % 4);
      if (st_tile)
      {
        ST4_TILE_IMPL(2, limit4)
        ST_TILE_REM_IMPL(2, limit)
      }
      else
      {
        ST4_TILE_IMPL(0, limit4)
        ST_TILE_REM_IMPL(0, limit);
      }
    }
  }

  // The DO_ADD template is used for u8 input - the add_val we add to convert the u8 to i8.
  template<bool DO_ADD>
  static void transpose_32_sme_impl(const void *A, void *T, transpose_idx_t M_bytes, transpose_idx_t N, transpose_idx_t LDA_bytes, transpose_idx_t LDT, int8_t add_value, char  zero_value)
  {
    uint32_t ld_tile = 0;
    transpose_idx_t ELT_CNTW = svcntw();
    transpose_idx_t ELT_CNTB = ELT_CNTW << 2;
    const transpose_idx_t ELT_CNTB2 = ELT_CNTB * 2;
    transpose_idx_t LDT_bytes = LDT << 2;
    void *T_ptr = NULL;
    transpose_idx_t i = 0;
    transpose_idx_t st_len = 0;
    transpose_idx_t st_count = 0;
    bool need_zero_fixup = (M_bytes % 4) != 0 && zero_value != 0;

    if (DO_ADD || need_zero_fixup) {
      __asm__ __volatile__("dup z4.b, %w[add_value]  \n\t"
                           "ptrue p1.b \n\t"
                           : : [add_value] "r" (add_value) : "memory", "cc");
    }
    // The kernel code assumes the inputs are multiple of 4.
    // Thus if the true dimension is not multiple of 4 it needs to be set to the zero value + add_value.
    // This is only done on the single vector loop remainder below this pair-vector loop.
    transpose_idx_t pair_threshold =  ((M_bytes % 4) != 0 && (zero_value != 0 || DO_ADD))  ? ELT_CNTB2 : ELT_CNTB + 1;
    for (; i + pair_threshold <= M_bytes; i += ELT_CNTB2)
    {
      __asm__ __volatile__("whilelt pn8.b, %[i], %[M_bytes], vlx2 \n\t"
                           :
                           : [M_bytes] "r"(M_bytes), [i] "r"(i)
                           : "memory", "cc");
      for (transpose_idx_t j = 0; j < N; j += ELT_CNTW)
      {
        const void *src_ptr = ((uint8_t *)A) + (j * LDA_bytes) + i;
        transpose_idx_t n = ARM64_SME_MIN(N - j, ELT_CNTW);
        register uint32_t za_index asm("w12");
        za_index = ld_tile;
        while (n >= 2)
        {
          __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add     %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z2.b, z3.b}, pn8/z, [%[src_ptr]] \n\t"
                               "add     %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               : [src_ptr] "+r"(src_ptr) : [LDA_bytes] "r"(LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__("add     { z0.b, z1.b }, { z0.b, z1.b }, z4.b \n\t"
                                 "add     { z2.b, z3.b }, { z2.b, z3.b }, z4.b \n\t"
                                 :::"memory");
          }
          __asm__ __volatile__("mova    za0h.b[%w[za_index], 0:1], { z0.b, z1.b } \n\t"
                               "mova    za0h.b[%w[za_index], 4:5], { z2.b, z3.b } \n\t"
                               : : [za_index] "r"(za_index)  : "memory");
          za_index += 8;
          n -= 2;
        }
        if (n)
        {
          if (DO_ADD) {
            __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn8/z, [%[src_ptr]] \n\t"
                                 "add     { z0.b, z1.b }, { z0.b, z1.b }, z4.b \n\t"
                                 "mova    za0h.b[%w[za_index], 0:1], { z0.b, z1.b } \n\t"
                                 :
                                 : [za_index] "r"(za_index), [src_ptr] "r"(src_ptr)
                                 : "memory");
          }
          else {
            __asm__ __volatile__("ld1b    { z0.b, z1.b }, pn8/z, [%[src_ptr]] \n\t"
                                 "mova    za0h.b[%w[za_index], 0:1], { z0.b, z1.b } \n\t"
                                 :
                                 : [za_index] "r"(za_index), [src_ptr] "r"(src_ptr)
                                 : "memory");
          }
        }
        ld_tile ^= 2;
        if (T_ptr)
        {
          sme_tranpose_32_st(T_ptr, LDT_bytes, ld_tile, st_len, st_count);
        }
        T_ptr = ((uint8_t *)T) + i * LDT + (j << 2);
        st_len = N - j;
        st_count = (M_bytes + 3 - i) >> 2;
      }
    }
    for ( ; i < M_bytes; i+= ELT_CNTB)
    {
      __asm__ __volatile__("whilelt p2.b, %[i], %[M_bytes] \n\t"
                           :
                           : [i] "r"(i), [M_bytes] "r"(M_bytes)
                           : "memory", "cc");
      
      bool need_zero_fixup_this_iteration = need_zero_fixup && (i + ELT_CNTB)  >= M_bytes;
      if (need_zero_fixup_this_iteration) {
        __asm__ __volatile__("dup z5.b, %w[zero_value] \n\t"
                             "not p4.b, p1/z, p2.b \n\t"
                             "add z4.b, p4/m, z4.b, z5.b \n\t"
                             :
                             : [i] "r"(i), [zero_value] "r"(zero_value)
                             : "memory");
      }
      for (transpose_idx_t j = 0; j < N; j += ELT_CNTW)
      {
        const void *src_ptr = ((uint8_t *)A) + (j * LDA_bytes) + i;
        transpose_idx_t n = ARM64_SME_MIN(N - j, ELT_CNTW);
        register uint32_t za_index asm("w12");
        za_index = ld_tile;
        if (DO_ADD || need_zero_fixup_this_iteration) {
          while (n >= 2)
          {
            __asm__ __volatile__("ld1b    z0.b, p2/z, [%[src_ptr]] \n\t"
                                 "add     %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    z1.b, p2/z, [%[src_ptr]] \n\t"
                                 "add     %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "add     { z0.b, z1.b},  { z0.b, z1.b}, z4.b \n\t"
                                 "mova    za0h.b[%w[za_index], 0], p1/m, z0.b \n\t"
                                 "mova    za0h.b[%w[za_index], 4], p1/m, z1.b \n\t"
                                 : [za_index] "+r"(za_index), [src_ptr] "+r"(src_ptr)
                                 : [LDA_bytes] "r"(LDA_bytes)
                                 : "memory");
            za_index += 8;
            n -= 2;
          }
          if (n)
          {
            __asm__ __volatile__("ld1b    z0.b, p2/z, [%[src_ptr]] \n\t"
                                 "add     z0.b, z0.b, z4.b \n\t"
                                 "mova    za0h.b[%w[za_index], 0], p1/m, z0.b \n\t"
                                 : : [za_index] "r"(za_index), [src_ptr] "r"(src_ptr) : "memory");
          }
        }
        else {
          while (n >= 2)
          {
            __asm__ __volatile__("ld1b    za0h.b[%w[za_index], 0], p2/z, [%[src_ptr]] \n\t"
                                 "add     %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    za0h.b[%w[za_index], 4], p2/z, [%[src_ptr]] \n\t"
                                 "add     %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 : [za_index] "+r"(za_index), [src_ptr] "+r"(src_ptr)
                                 : [LDA_bytes] "r"(LDA_bytes)
                                 : "memory");
            za_index += 8;
            n -= 2;
          }
          if (n)
          {
            __asm__ __volatile__("ld1b    za0h.b[%w[za_index], 0], p2/z, [%[src_ptr]] \n\t"
                                 : : [za_index] "r"(za_index), [src_ptr] "r"(src_ptr) : "memory");
          }
        }
        ld_tile ^= 2;
        if (T_ptr)
        {
          sme_tranpose_32_st(T_ptr, LDT_bytes, ld_tile, st_len, st_count);
        }
        T_ptr = ((uint8_t *)T) + i * LDT + (j << 2);
        st_len = N - j;
        st_count = (M_bytes + 3 - i) >> 2;
      }
    }
    sme_tranpose_32_st(T_ptr, LDT_bytes, ld_tile ^ 2, st_len, st_count);
  }

  static void transpose_32_sme_add(const void *A, void *T, transpose_idx_t M_bytes, transpose_idx_t N, transpose_idx_t LDA_bytes,
                                   transpose_idx_t LDT, unsigned char add_val, char zero_value)
  {
    uint8_t abi_stack[64] __attribute__((aligned(16)));
    SMSTART(abi_stack);
    if (add_val) {
      transpose_32_sme_impl<true>(A, T, M_bytes, N, LDA_bytes, LDT, add_val, zero_value);
    }
    else
    {
      transpose_32_sme_impl<false>(A, T, M_bytes, N, LDA_bytes, LDT, 0, zero_value);
    }
    SMSTOP(abi_stack);
  }

  static void memcpy_sme_impl(transpose_idx_t N, const void *src_ptr_void, void* dst_ptr_void)
  {
    const char *src_ptr = (char *)src_ptr_void;
    char *dst_ptr = (char *)dst_ptr_void;
    const char *src_end = src_ptr + N;
    asm volatile("ptrue pn8.b" : : : "memory", "cc");

    transpose_idx_t  elements_count16 = svcntb() * 16;
    transpose_idx_t N16 = N - (N % elements_count16);
    const char* src_end16 = src_ptr + N16;
    
    if (N16) {
      __asm__ __volatile__ ("ld1b  { z0.b, z1.b, z2.b, z3.b }, pn8/z, [%[src_ptr], #0, mul vl] \n\t"
                            "ld1b  { z4.b, z5.b, z6.b, z7.b }, pn8/z, [%[src_ptr], #4, mul vl] \n\t"
                            "ld1b  { z8.b, z9.b, z10.b, z11.b }, pn8/z, [%[src_ptr], #8, mul vl] \n\t"
                            "ld1b  { z12.b, z13.b, z14.b, z15.b }, pn8/z, [%[src_ptr], #12, mul vl] \n\t"
                            : : [src_ptr] "r" (src_ptr) : "memory");
      src_ptr += elements_count16;
    }
    for ( ; src_ptr < src_end16; src_ptr += elements_count16, dst_ptr += elements_count16) {
      __asm__ __volatile__ ("st1b  { z0.b, z1.b, z2.b, z3.b }, pn8, [%[dst_ptr], #0, mul vl] \n\t"
                            "ld1b  { z0.b, z1.b, z2.b, z3.b }, pn8/z, [%[src_ptr], #0, mul vl] \n\t"
                            "st1b  { z4.b, z5.b, z6.b, z7.b }, pn8, [%[dst_ptr], #4, mul vl] \n\t"
                            "ld1b  { z4.b, z5.b, z6.b, z7.b }, pn8/z, [%[src_ptr], #4, mul vl] \n\t"
                            "st1b  { z8.b, z9.b, z10.b, z11.b }, pn8, [%[dst_ptr], #8, mul vl] \n\t"
                            "ld1b  { z8.b, z9.b, z10.b, z11.b }, pn8/z, [%[src_ptr], #8, mul vl] \n\t"
                            "st1b  { z12.b, z13.b, z14.b, z15.b }, pn8, [%[dst_ptr], #12, mul vl] \n\t"
                            "ld1b  { z12.b, z13.b, z14.b, z15.b }, pn8/z, [%[src_ptr], #12, mul vl] \n\t"
                            :
                            : [src_ptr] "r" (src_ptr), [dst_ptr] "r" (dst_ptr)
                            : "memory");
    }
    if (N16) {
      __asm__ __volatile__ ("st1b  { z0.b, z1.b, z2.b, z3.b }, pn8, [%[dst_ptr], #0, mul vl] \n\t"
                            "st1b  { z4.b, z5.b, z6.b, z7.b }, pn8, [%[dst_ptr], #4, mul vl] \n\t"
                            "st1b  { z8.b, z9.b, z10.b, z11.b }, pn8, [%[dst_ptr], #8, mul vl] \n\t"
                            "st1b  { z12.b, z13.b, z14.b, z15.b }, pn8, [%[dst_ptr], #12, mul vl] \n\t"
                            : : [dst_ptr] "r" (dst_ptr) : "memory");
      dst_ptr += elements_count16;
    }

    transpose_idx_t  elements_count4  = svcntb() * 4;
    while (src_ptr + elements_count4 <= src_end) {
      __asm__ __volatile__ ("ld1b  { z0.b, z1.b, z2.b, z3.b }, pn8/z, [%[src_ptr]] \n\t"
                            "st1b  { z0.b, z1.b, z2.b, z3.b }, pn8, [%[dst_ptr]] \n\t"
                            "incb %[src_ptr], all, mul #4  \n\t"
                            "incb %[dst_ptr], all, mul #4  \n\t"
                            : [src_ptr] "+r" (src_ptr), [dst_ptr] "+r" (dst_ptr)
                            :
                            : "memory");
    }
    while ( src_ptr <  src_end ) {
      __asm__ __volatile__ ("whilelt pn8.b, %[src_ptr], %[src_end], vlx4 \n\t"
                            "ld1b  { z0.b, z1.b, z2.b, z3.b }, pn8/z, [%[src_ptr]] \n\t"
                            "st1b  { z0.b, z1.b, z2.b, z3.b }, pn8, [%[dst_ptr]] \n\t"
                            "incb %[src_ptr], all, mul #4  \n\t"
                            "incb %[dst_ptr], all, mul #4  \n\t"
                            : [src_ptr] "+r" (src_ptr), [dst_ptr] "+r" (dst_ptr)
                            : [src_end] "r" (src_end)
                            : "memory", "cc" );
    }
  }
 
  template<bool DO_ADD, bool LESS_THAN_4>
  static void row_major_pack_batch(const int8_t *src_ptr_base, int8_t *rhs_ptr, transpose_idx_t num_rows,  transpose_idx_t num_elts, transpose_idx_t LDA, int8_t add_value, char src_zero_point) {
    transpose_idx_t elements_count = svcntb();
    transpose_idx_t elements_count2 = elements_count * 2;
    transpose_idx_t elements_count3 = elements_count * 3;
    transpose_idx_t elements_count4 = elements_count * 4;
    transpose_idx_t elements_count8 = elements_count * 8;
    transpose_idx_t elements_count16 = elements_count * 16;
    transpose_idx_t LDA_bytes = LDA;

    static_assert(transpose_idx_t(-1) < 0, "code below assumes that negative numbers exist.");
    __asm__ __volatile__("dup z8.b, #1" : : : "memory");
    if (DO_ADD) {
      __asm__ __volatile__("dup z4.b, %w[add_value] " : : [add_value] "r" (add_value) : "memory");
    }

    

    if (LESS_THAN_4) {
      int8_t edge_value = src_zero_point + add_value;
      __asm__ __volatile__("dup z19.b, %w[edge_value] \n\t"
                           "dup z23.b, %w[edge_value] \n\t"
                           "dup z27.b, %w[edge_value] \n\t"
                           "dup z31.b, %w[edge_value] \n\t"
                           : : [edge_value] "r" (edge_value) : "memory");
      if (num_rows < 3) {
        __asm__ __volatile__("dup z18.b, %w[edge_value] \n\t"
                             "dup z22.b, %w[edge_value] \n\t"
                             "dup z26.b, %w[edge_value] \n\t"
                             "dup z30.b, %w[edge_value] \n\t"
                             : : [edge_value] "r" (edge_value) : "memory");
        if (num_rows < 2) {
          __asm__ __volatile__("dup z17.b, %w[edge_value] \n\t"
                               "dup z21.b, %w[edge_value] \n\t"
                               "dup z25.b, %w[edge_value] \n\t"
                               "dup z29.b, %w[edge_value] \n\t"
                               : : [edge_value] "r" (edge_value) : "memory");
        }
      }
    }
    while (num_elts > elements_count3){
      const int8_t *src_ptr = src_ptr_base;
      __asm__ __volatile__  ("whilelt pn8.b, xzr, %[num_elts], vlx4 \n\t"
                             "pext    { p0.b, p1.b }, pn8[0] \n\t"
                             "pext    { p2.b, p3.b }, pn8[1] \n\t"
                             : : [num_elts] "r" (num_elts) : "memory", "cc");
      if (LESS_THAN_4) {
        __asm__ __volatile__  ("ld1b    { z16.b, z20.b, z24.b, z28.b }, pn8/z, [%[src_ptr]] \n\t" // must read first
                               : :  [src_ptr] "r" (src_ptr)  : "memory");
        if (num_rows == 1) {
          if (DO_ADD) {
            __asm__ __volatile__  ("add   z16.b, z16.b, z4.b \n\t"
                                   "add   z20.b, z20.b, z4.b \n\t"
                                   "add   z24.b, z24.b, z4.b \n\t"
                                   "add   z28.b, z28.b, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
        else if (num_rows == 2) {
          __asm__ __volatile__  ("add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z17.b, z21.b, z25.b, z29.b }, pn8/z, [%[src_ptr]] \n\t"
                                 : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__  ("add   {z16.b, z17.b}, {z16.b, z17.b}, z4.b \n\t"
                                   "add   {z20.b, z21.b}, {z20.b, z21.b}, z4.b \n\t"
                                   "add   {z24.b, z25.b}, {z24.b, z25.b}, z4.b \n\t"
                                   "add   {z28.b, z29.b}, {z28.b, z29.b}, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
        else { // num_rows == 3
          __asm__ __volatile__  ("add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z17.b, z21.b, z25.b, z29.b }, pn8/z, [%[src_ptr]] \n\t"
                                 "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z18.b, z22.b, z26.b, z30.b }, pn8/z, [%[src_ptr]] \n\t"
                                 : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__  ("add   {z16.b, z17.b}, {z16.b, z17.b}, z4.b \n\t"
                                   "add   z18.b, z18.b, z4.b \n\t"
                                   "add   {z20.b, z21.b}, {z20.b, z21.b}, z4.b \n\t"
                                   "add   z22.b, z22.b, z4.b \n\t"
                                   "add   {z24.b, z25.b}, {z24.b, z25.b}, z4.b \n\t"
                                   "add   z26.b, z26.b, z4.b \n\t"
                                   "add   {z28.b, z29.b}, {z28.b, z29.b}, z4.b \n\t"
                                   "add   z30.b, z30.b, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
      }
      else {
        __asm__ __volatile__  ("ld1b    { z16.b, z20.b, z24.b, z28.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z17.b, z21.b, z25.b, z29.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z18.b, z22.b, z26.b, z30.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z19.b, z23.b, z27.b, z31.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
        //if needed add flip the top bit.
        if (DO_ADD) {
          __asm__ __volatile__  ("add   { z16.b, z17.b, z18.b, z19.b }, { z16.b, z17.b, z18.b, z19.b }, z4.b \n\t"
                                 "add   { z20.b, z21.b, z22.b, z23.b }, { z20.b, z21.b, z22.b, z23.b }, z4.b \n\t"
                                 "add   { z24.b, z25.b, z26.b, z27.b }, { z24.b, z25.b, z26.b, z27.b }, z4.b \n\t"
                                 "add   { z28.b, z29.b, z30.b, z31.b }, { z28.b, z29.b, z30.b, z31.b }, z4.b \n\t"
                                 :  :  : "memory");
        }
      }
      __asm__ __volatile__  ("st4b    { z16.b, z17.b, z18.b, z19.b }, p0, [%[rhs_ptr]] \n\t"
                             "st4b    { z20.b, z21.b, z22.b, z23.b }, p1, [%[rhs_ptr], #4, mul vl] \n\t"
                             "st4b    { z24.b, z25.b, z26.b, z27.b }, p2, [%[rhs_ptr], #8, mul vl] \n\t"
                             "st4b    { z28.b, z29.b, z30.b, z31.b }, p3, [%[rhs_ptr], #12, mul vl] \n\t"
                             :  : [rhs_ptr] "r" (rhs_ptr) : "memory");
      num_elts -= elements_count4;
      rhs_ptr += elements_count16;
      src_ptr_base += elements_count4;
    }
    while (num_elts > elements_count){
      const int8_t *src_ptr = src_ptr_base;
      __asm__ __volatile__  ("whilelt pn8.b, xzr, %[num_elts], vlx2 \n\t"
                             "pext    { p0.b, p1.b }, pn8[0] \n\t"
                             : : [num_elts] "r" (num_elts) : "memory", "cc");
      if (LESS_THAN_4) {
        __asm__ __volatile__  ("ld1b    { z16.b, z24.b }, pn8/z, [%[src_ptr]] \n\t" // must read first
                               : :  [src_ptr] "r" (src_ptr)  : "memory");
        if (num_rows == 1) {
          if (DO_ADD) {
            __asm__ __volatile__  ("add   z16.b, z16.b, z4.b \n\t"
                                   "add   z24.b, z24.b, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
        else if (num_rows == 2) {
          __asm__ __volatile__  ("add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z17.b, z25.b }, pn8/z, [%[src_ptr]] \n\t"
                                 : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__  ("add   {z16.b, z17.b}, {z16.b, z17.b}, z4.b \n\t"
                                   "add   {z24.b, z25.b}, {z24.b, z25.b}, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
        else { // num_rows == 3
          __asm__ __volatile__  ("add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z17.b, z25.b }, pn8/z, [%[src_ptr]] \n\t"
                                 "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z18.b, z26.b }, pn8/z, [%[src_ptr]] \n\t"
                                 : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__  ("add   {z16.b, z17.b}, {z16.b, z17.b}, z4.b \n\t"
                                   "add   z18.b, z18.b, z4.b \n\t"
                                   "add   {z24.b, z25.b}, {z24.b, z25.b}, z4.b \n\t"
                                   "add   z26.b, z26.b, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
      }
      else {
        __asm__ __volatile__  ("ld1b    { z16.b, z24.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z17.b, z25.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z18.b, z26.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z19.b, z27.b }, pn8/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
        //if needed add flip the top bit.
        if (DO_ADD) {
          __asm__ __volatile__  ("add   { z16.b, z17.b, z18.b, z19.b }, { z16.b, z17.b, z18.b, z19.b }, z4.b \n\t"
                                 "add   { z24.b, z25.b, z26.b, z27.b }, { z24.b, z25.b, z26.b, z27.b }, z4.b \n\t"
                                 :  :  : "memory");
        }
      }
      __asm__ __volatile__  ("st4b    { z16.b, z17.b, z18.b, z19.b }, p0, [%[rhs_ptr]] \n\t"
                             "st4b    { z24.b, z25.b, z26.b, z27.b }, p1, [%[rhs_ptr], #4, mul vl] \n\t"
                             :  : [rhs_ptr] "r" (rhs_ptr) : "memory");
      num_elts -= elements_count2;
      rhs_ptr += elements_count8;
      src_ptr_base += elements_count2;
    }
    while (num_elts > 0){
      const int8_t *src_ptr = src_ptr_base;
      __asm__ __volatile__  ("whilelt p0.b, xzr, %[num_elts] \n\t"
                             : : [num_elts] "r" (num_elts) : "memory", "cc");
      if (LESS_THAN_4) {
        __asm__ __volatile__  ("ld1b    { z16.b }, p0/z, [%[src_ptr]] \n\t" // must read first
                               : :  [src_ptr] "r" (src_ptr)  : "memory");
        if (num_rows == 1) {
          if (DO_ADD) {
            __asm__ __volatile__  ("add   z16.b, z16.b, z4.b" :  :  : "memory");
          }
        }
        else if (num_rows == 2) {
          __asm__ __volatile__  ("add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z17.b }, p0/z, [%[src_ptr]] \n\t"
                                 : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__  ("add   {z16.b, z17.b}, {z16.b, z17.b}, z4.b" :  :  : "memory");
          }
        }
        else { // num_rows == 3
          __asm__ __volatile__  ("add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z17.b }, p0/z, [%[src_ptr]] \n\t"
                                 "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                                 "ld1b    { z18.b }, p0/z, [%[src_ptr]] \n\t"
                                 : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
          if (DO_ADD) {
            __asm__ __volatile__  ("add   {z16.b, z17.b}, {z16.b, z17.b}, z4.b \n\t"
                                   "add   z18.b, z18.b, z4.b \n\t"
                                   :  :  : "memory");
          }
        }
      }
      else {
        __asm__ __volatile__  ("ld1b    { z16.b }, p0/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z17.b }, p0/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z18.b }, p0/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               "ld1b    { z19.b }, p0/z, [%[src_ptr]] \n\t"
                               "add   %[src_ptr], %[src_ptr], %[LDA_bytes] \n\t"
                               : [src_ptr] "+r" (src_ptr) : [LDA_bytes] "r" (LDA_bytes) : "memory");
        //if needed add flip the top bit.
        if (DO_ADD) {
          __asm__ __volatile__  ("add   { z16.b, z17.b, z18.b, z19.b }, { z16.b, z17.b, z18.b, z19.b }, z4.b " :  :  : "memory");
        }
      }
      __asm__ __volatile__  ("st4b    { z16.b, z17.b, z18.b, z19.b }, p0, [%[rhs_ptr]]" :  : [rhs_ptr] "r" (rhs_ptr) : "memory");
      num_elts -= elements_count;
      rhs_ptr += elements_count4;
      src_ptr_base += elements_count;
    }
  }
  
  static void PackInt8RowMajorForSmeImpl(const int8_t *src_ptr, int8_t *packed_ptr, int src_rows, int depth, int src_stride, int packed_stride, char add_val, char src_zero_point)
  {
    uint8_t abi_stack[64] __attribute__((aligned(16)));
    SMSTART(abi_stack);
    int row = 0;
    for (; row + 4 <= depth; row += 4, src_ptr += (src_stride * 4), packed_ptr += (packed_stride *  4)) {
      if (add_val) {
        row_major_pack_batch<true, false>(src_ptr, packed_ptr, 4, src_rows, src_stride, add_val, src_zero_point);
      }
      else {
        row_major_pack_batch<false, false>(src_ptr, packed_ptr, 4, src_rows, src_stride, add_val, src_zero_point);
      }
    }
    if ( row < depth ) {
      if (add_val) {
        row_major_pack_batch<true, true>(src_ptr, packed_ptr, depth - row, src_rows, src_stride, add_val, src_zero_point);
      }
      else {
        row_major_pack_batch<false, true>(src_ptr, packed_ptr, depth - row, src_rows, src_stride, add_val, src_zero_point);
      }
    }
    SMSTOP(abi_stack);
  }

  // Entry functions for the SGEMM packing functions.
  void PackFloatColMajorForSme(const float *src_ptr, float *packed_ptr, int src_cols, int depth, int src_stride, int packed_stride)
  {
    transpose_32_sme_add(src_ptr, packed_ptr, depth << 2, src_cols, src_stride << 2, packed_stride, 0, 0);
  }

  void PackFloatRowMajorForSme(const float *src_ptr, float *packed_ptr, int src_rows, int depth, int src_stride, int packed_stride)
  {
    uint8_t abi_stack[64] __attribute__((aligned(16)));
    SMSTART(abi_stack);
    for (int j = 0; j < depth; j++, src_ptr += src_stride, packed_ptr += packed_stride)
    {
      memcpy_sme_impl(sizeof(float) * src_rows, src_ptr, packed_ptr);
    }
    SMSTOP(abi_stack);
  }

  // Entry functions for the GEMMi8 packing functions.
  // The calls to gemvi8_packed_sme are for calculating the rows/cols sums.
  void PackInt8ColMajorForSme(const int8_t *src_ptr, int8_t *packed_ptr, int32_t *sums_ptr, int src_cols, int depth, int packed_depth, int src_stride, int packed_stride, char add_val, char src_zero_point)
  {
    transpose_32_sme_add(src_ptr, packed_ptr, depth, src_cols, src_stride, packed_stride, add_val, src_zero_point);
    gemvi8_packed_sme(src_cols, packed_depth, (int32_t *)packed_ptr, packed_stride, sums_ptr, false);
  }

  void PackInt8RowMajorForSme(const int8_t *src_ptr, int8_t *packed_ptr, int32_t *sums_ptr, int src_rows, int depth, int packed_depth, int src_stride, int packed_stride, char add_val, char src_zero_point)
  {
    PackInt8RowMajorForSmeImpl(src_ptr, packed_ptr, src_rows, depth, src_stride, packed_stride, add_val, src_zero_point);
    gemvi8_packed_sme(src_rows, packed_depth, (int32_t *)packed_ptr, packed_stride, sums_ptr, false);
  }

} // namespace ruy

#endif // RUY_PLATFORM_ARM64_SME && RUY_OPT(ASM)
