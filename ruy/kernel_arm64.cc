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

namespace ruy {

// MSVC on Windows ARM64 does not support AT&T-style GCC inline assembly
// (asm volatile). The asm-based kernels below are guarded out under MSVC;
// MSVC-compatible NEON intrinsic kernels follow after the closing #endif.
#if RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && !defined(_MSC_VER)

#define RUY_ASM_LABEL_STORE_UINT8 91
#define RUY_ASM_LABEL_STORE_INT8 92
#define RUY_ASM_LABEL_STORE_INT16 93
#define RUY_ASM_LABEL_STORE_INT32 94
#define RUY_ASM_LABEL_AFTER_STORE 99

#define RUY_OFFSET_BIAS 0
#define RUY_OFFSET_LHS_SUMS 8
#define RUY_OFFSET_RHS_SUMS 16
#define RUY_OFFSET_LHS_BASE_PTR 24
#define RUY_OFFSET_MULTIPLIER_FIXEDPOINT 32
#define RUY_OFFSET_MULTIPLIER_EXPONENT 40
#define RUY_OFFSET_RHS_BASE_PTR 48
#define RUY_OFFSET_DST_BASE_PTR 56
#define RUY_OFFSET_LHS_ZERO_POINT 64
#define RUY_OFFSET_RHS_ZERO_POINT 68
#define RUY_OFFSET_DST_ZERO_POINT 72
#define RUY_OFFSET_PROD_ZP_DEPTH 76
#define RUY_OFFSET_START_ROW 80
#define RUY_OFFSET_START_COL 84
#define RUY_OFFSET_LAST_ROW 88
#define RUY_OFFSET_LAST_COL 92
#define RUY_OFFSET_DST_ROWS 96
#define RUY_OFFSET_DST_COLS 100
#define RUY_OFFSET_LHS_STRIDE 104
#define RUY_OFFSET_RHS_STRIDE 108
#define RUY_OFFSET_DST_STRIDE 112
#define RUY_OFFSET_DEPTH 116
#define RUY_OFFSET_CLAMP_MIN 120
#define RUY_OFFSET_CLAMP_MAX 124
#define RUY_OFFSET_FLAGS 128

template <typename Params>
void CheckOffsetsInKernelParams8bit(const Params&) {
  static_assert(offsetof(Params, lhs_zero_point) == RUY_OFFSET_LHS_ZERO_POINT,
                "");
  static_assert(offsetof(Params, rhs_zero_point) == RUY_OFFSET_RHS_ZERO_POINT,
                "");
  static_assert(offsetof(Params, dst_zero_point) == RUY_OFFSET_DST_ZERO_POINT,
                "");
  static_assert(offsetof(Params, prod_zp_depth) == RUY_OFFSET_PROD_ZP_DEPTH,
                "");
  static_assert(offsetof(Params, multiplier_fixedpoint) ==
                    RUY_OFFSET_MULTIPLIER_FIXEDPOINT,
                "");
  static_assert(
      offsetof(Params, multiplier_exponent) == RUY_OFFSET_MULTIPLIER_EXPONENT,
      "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(offsetof(Params, bias) == RUY_OFFSET_BIAS, "");
  static_assert(offsetof(Params, lhs_sums) == RUY_OFFSET_LHS_SUMS, "");
  static_assert(offsetof(Params, rhs_sums) == RUY_OFFSET_RHS_SUMS, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
}

// Fast-int8-trick kernel, similar to this production gemmlowp kernel:
// NEON_64bit_GEMM_Int8Operands_AccumTwoWithin16Bits
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L2296
//
// Relevant target CPUs for this kernel include ARM Cortex-A73 and Cortex-A75,
// since these are 64-bit, out-of-order and without dotprod support.
void Kernel8bitNeon(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon)");
  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // v4 -- v7 from RHS:
  //
  //                                      int8 RHS 16x4 block
  //                           /-----------------------------------------|
  //                           |v4.b[0]          ...           v7.b[0]   |
  //                           |  ...                            ...     |
  //                           |v4.b[15]         ...           v7.b[15]  |
  //                           \-----------------------------------------/
  //    int8 LHS 4x16 block
  //  /---------------------\  /-----------------------------------------|
  //  |v0.b[0] ... v0.b[15] |  |v16.4s           ...           v28.4s    |
  //  |v1.b[0] ... v1.b[15] |  |v17.4s           ...           v29.4s    |
  //  |v2.b[0] ... v2.b[15] |  |v18.4s           ...           v30.4s    |
  //  |v3.b[0] ... v3.b[15] |  |v19.4s           ...           v31.4s    |
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 4x4 block
  //
  // No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
  // optimization for this kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "sadalp  v24.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "sadalp  v25.4s, v9.8h\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "sadalp  v26.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "sadalp  v27.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "sadalp  v28.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "sadalp  v29.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "sadalp  v30.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"



        // Each iteration of this loop advances by 16 levels of depth.
        "add w1, w1, #16\n"

        // Loop termination condition
        "cmp w1, w12\n"

        "blt 2b\n"

        "79:\n"

        "sadalp  v16.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "sadalp  v25.4s, v9.8h\n"
        "sadalp  v26.4s, v10.8h\n"
        "sadalp  v27.4s, v11.8h\n"
        "sadalp  v28.4s, v12.8h\n"
        "sadalp  v29.4s, v13.8h\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 4x4 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x4 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v17.4s\n"
        "addp v18.4s, v18.4s, v19.4s\n"
        "addp v20.4s, v20.4s, v21.4s\n"
        "addp v22.4s, v22.4s, v23.4s\n"
        "addp v24.4s, v24.4s, v25.4s\n"
        "addp v26.4s, v26.4s, v27.4s\n"
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v16.4s, v16.4s, v18.4s\n"
        "addp v17.4s, v20.4s, v22.4s\n"
        "addp v18.4s, v24.4s, v26.4s\n"
        "addp v19.4s, v28.4s, v30.4s\n"

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec

        // Now we load: bias data, LHS sums data, RHS sums data.

        // First, load the base pointers from the params.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 4 bias values.
        "ld1 {v14.4s}, [x1]\n"

        // Load the multiplier_fixedpoint values.
        "add x5, x4, x3, lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"
        "ld1 {v15.4s}, [x4]\n" // multiplier_fixedpoint

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v14.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v14.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v20.4s, v14.s[0]\n"
        "dup v21.4s, v14.s[1]\n"
        "dup v22.4s, v14.s[2]\n"
        "dup v23.4s, v14.s[3]\n"
        "add v16.4s, v16.4s, v20.4s\n"
        "add v17.4s, v17.4s, v21.4s\n"
        "add v18.4s, v18.4s, v22.4s\n"
        "add v19.4s, v19.4s, v23.4s\n"
        "7:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[1]\n"
        "mls v18.4s, v10.4s, v14.s[2]\n"
        "mls v19.4s, v10.4s, v14.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v11.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v11.4s\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, x3, lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ld1 {v14.4s}, [x1]\n"

        "smin v11.4s, v8.4s, v14.4s\n"
        "sub v12.4s, v14.4s, v11.4s\n"

        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 8f\n"
        // Case where channels are rows

        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v12.4s\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "sshl v18.4s, v18.4s, v12.4s\n"
        "sshl v19.4s, v19.4s, v12.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v15.4s\n"
        "sqdmulh v17.4s, v17.4s, v15.4s\n"
        "sqdmulh v18.4s, v18.4s, v15.4s\n"
        "sqdmulh v19.4s, v19.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v11.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v11.4s\n"
        "b 9f\n"

        "8:\n"
        // Case where channels are columns

        // Apply the positive exponent part of the multiplier.
        "dup v20.4s, v12.s[0]\n"
        "dup v21.4s, v12.s[1]\n"
        "dup v22.4s, v12.s[2]\n"
        "dup v23.4s, v12.s[3]\n"
        "sshl v16.4s, v16.4s, v20.4s\n"
        "sshl v17.4s, v17.4s, v21.4s\n"
        "sshl v18.4s, v18.4s, v22.4s\n"
        "sshl v19.4s, v19.4s, v23.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v15.s[0]\n"
        "sqdmulh v17.4s, v17.4s, v15.s[1]\n"
        "sqdmulh v18.4s, v18.4s, v15.s[2]\n"
        "sqdmulh v19.4s, v19.4s, v15.s[3]\n"

        // Apply the negative exponent part of the multiplier.
        "dup v20.4s, v11.s[0]\n"
        "dup v21.4s, v11.s[1]\n"
        "dup v22.4s, v11.s[2]\n"
        "dup v23.4s, v11.s[3]\n"
        "srshl v16.4s, v16.4s, v20.4s\n"
        "srshl v17.4s, v17.4s, v21.4s\n"
        "srshl v18.4s, v18.4s, v22.4s\n"
        "srshl v19.4s, v19.4s, v23.4s\n"
        "9:\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "sqxtun2 v16.16b, v17.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"

        // Cast-and-saturate from int16 to int8
        "sqxtn v16.8b, v16.8h\n"
        "sqxtn2 v16.16b, v17.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.4h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[1], [x3], #2\n"
        "st1 {v16.h}[2], [x3], #2\n"
        "st1 {v16.h}[3], [x3], #2\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[5], [x3], #2\n"
        "st1 {v16.h}[6], [x3], #2\n"
        "st1 {v16.h}[7], [x3], #2\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v17.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[1], [x3], #2\n"
        "st1 {v17.h}[2], [x3], #2\n"
        "st1 {v17.h}[3], [x3], #2\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v17.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[5], [x3], #2\n"
        "st1 {v17.h}[6], [x3], #2\n"
        "st1 {v17.h}[7], [x3], #2\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #8\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // At this point, v20 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        "str q18, [%[dst_tmp_buf], #32]\n"
        "str q19, [%[dst_tmp_buf], #48]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v16.s}[1], [x3], #4\n"
        "st1 {v16.s}[2], [x3], #4\n"
        "st1 {v16.s}[3], [x3], #4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v17.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v17.s}[1], [x3], #4\n"
        "st1 {v17.s}[2], [x3], #4\n"
        "st1 {v17.s}[3], [x3], #4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v18.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v18.s}[1], [x3], #4\n"
        "st1 {v18.s}[2], [x3], #4\n"
        "st1 {v18.s}[3], [x3], #4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v19.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v19.s}[1], [x3], #4\n"
        "st1 {v19.s}[2], [x3], #4\n"
        "st1 {v19.s}[3], [x3], #4\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #16\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #4\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #4\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #16\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Similar to existing Kernel8bitNeon but specialized for the case of
// RHS cols == 1.
// Relevant target CPUs for this kernel include ARM Cortex-A73 and Cortex-A75,
// since these are 64-bit, out-of-order and without dotprod support.
void Kernel8bitNeon1Col(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  RUY_DCHECK(!(params.flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL));

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v19 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // v4 from RHS:
  //
  //                         int8 RHS 16x1 block
  //                           /-----------|
  //                           |v4.b[0]    |
  //                           |  ...      |
  //                           |v4.b[15]   |
  //                           \-----------/
  //    int8 LHS 4x16 block
  //  /---------------------\  /-----------|
  //  |v0.b[0] ... v0.b[15] |  |v16.4s     |
  //  |v1.b[0] ... v1.b[15] |  |v17.4s     |
  //  |v2.b[0] ... v2.b[15] |  |v18.4s     |
  //  |v3.b[0] ... v3.b[15] |  |v19.4s     |
  //  \---------------------/  \-----------/
  //                         int32 accumulators 4x1 block
  //
  // No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
  // optimization for this kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "add %[rhs_ptr], %[rhs_ptr], #48\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "add %[rhs_ptr], %[rhs_ptr], #48\n"
        "sadalp  v17.4s, v9.8h\n"
        "sadalp  v18.4s, v10.8h\n"
        "sadalp  v19.4s, v11.8h\n"

        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Each iteration of this loop advances by 16 levels of depth.
        "add w1, w1, #16\n"

        // Loop termination condition
        "cmp w1, w12\n"

        "blt 2b\n"

        "79:\n"

        "sadalp  v16.4s, v8.8h\n"
        "sadalp  v17.4s, v9.8h\n"
        "sadalp  v18.4s, v10.8h\n"
        "sadalp  v19.4s, v11.8h\n"

        // End of accumulation. The registers v16 -- v19 contain the final
        // int32 accumulator values of the current 4x1 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x1 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v17.4s\n"
        "addp v18.4s, v18.4s, v19.4s\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v16.4s, v16.4s, v18.4s\n"

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        // (still multiply column stride by 4 due to packing)
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ld1 {v15.4s}, [x4]\n" // multiplier_fixedpoint

        // Now we load: bias data, LHS sums data, RHS sums data.

        // First, load the base pointers from the params.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        "add x5, x1, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 4 bias values.
        "ld1 {v14.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "add %[rhs_ptr], %[rhs_ptr], #48\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // (all four 32-bit accumulators are in v16 at this point)
        "add v16.4s, v16.4s, v14.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ld1 {v14.4s}, [x1]\n"

        "smin v11.4s, v8.4s, v14.4s\n"
        "sub v12.4s, v14.4s, v11.4s\n"

        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v12.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this instruction, all data is in lower half (64-bits) of v16
        "sqxtn v16.4h, v16.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        // Now all data is in the first 32-bits of v16
        "sqxtun v16.8b, v16.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x1, there are some 4x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x1 block fit
        "csel w1, w1, w3, le\n"

        // Test if w1==4, i.e. if all of the 4x1 block fits.
        "cmp w1, w3\n"

        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x1 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x1 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in the lower half (64 bits) of v16.
        "sqxtn v16.4h, v16.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to int8
        "sqxtn v16.8b, v16.8h\n"
        // At this point, we only need 4 lowest 8-bit values in v16.

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"

        // Test if w1==4, i.e. if all of the 4x1 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.4h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        // After this instruction, all data is in lower half of v16.
        "sqxtn v16.4h, v16.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // Load the clamp_min, clamp_max bounds
        "ldrh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.h}[0], [x3], #2\n"
        "st1 {v16.h}[1], [x3], #2\n"
        "st1 {v16.h}[2], [x3], #2\n"
        "st1 {v16.h}[3], [x3], #2\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #8\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"

        // Test if w1==4 i.e. if all of the 4x1 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.s}[0], [x3], #4\n"
        "st1 {v16.s}[1], [x3], #4\n"
        "st1 {v16.s}[2], [x3], #4\n"
        "st1 {v16.s}[3], [x3], #4\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #16\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #4\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #4\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19");
}

// Variant of the above Kernel8bitNeon, tuned for A55-ish CPUs.
// Specifically here, the relevant in-order CPUs are ARM Cortex-A53 and
// the original Cortex-A55, since these are 64-bit and do not support dotprod.
//
// While this kernel does not have a direct equivalent in gemmlowp, it was
// developed based on insights that David Mansell at ARM shared with their
// contribution of gemmlowp kernels tuned for Cortex-A53, with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A53:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4215
void Kernel8bitNeonA55ish(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon, optimized for in-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // v4 -- v7 from RHS:
  //
  //                                      int8 RHS 16x4 block
  //                           /-----------------------------------------|
  //                           |v4.b[0]          ...           v7.b[0]   |
  //                           |  ...                            ...     |
  //                           |v4.b[15]         ...           v7.b[15]  |
  //                           \-----------------------------------------/
  //    int8 LHS 4x16 block
  //  /---------------------\  /-----------------------------------------|
  //  |v0.b[0] ... v0.b[15] |  |v16.4s           ...           v28.4s    |
  //  |v1.b[0] ... v1.b[15] |  |v17.4s           ...           v29.4s    |
  //  |v2.b[0] ... v2.b[15] |  |v18.4s           ...           v30.4s    |
  //  |v3.b[0] ... v3.b[15] |  |v19.4s           ...           v31.4s    |
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 4x4 block
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        RUY_MAKE_ZERO(v16)
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        RUY_MAKE_ZERO(v17)
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        RUY_MAKE_ZERO(v18)
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        RUY_MAKE_ZERO(v19)
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v20)
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v21)
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        RUY_MAKE_ZERO(v22)
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
        RUY_MAKE_ZERO(v23)

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v24)
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v25)
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v26)
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v27)
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v28)
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v29)
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v30)
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v31)


        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ldr d4, [%[rhs_ptr], #0]\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "ldr x7, [%[rhs_ptr], #8]\n"
        "sadalp  v17.4s, v9.8h\n"
        "ldr d5, [%[rhs_ptr], #16]\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "ldr x8, [%[rhs_ptr], #24]\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        "sadalp  v20.4s, v12.8h\n"
        // Each iteration of this loop advances by 16 levels of depth.
        "add w1, w1, #16\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        // Loop termination condition
        "cmp w1, w12\n"
        "sadalp  v21.4s, v13.8h\n"
        "ldr x3, [%[lhs_ptr], #-56]\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "ldr x4, [%[lhs_ptr], #-40]\n"
        "sadalp  v22.4s, v14.8h\n"
        "ldr x5, [%[lhs_ptr], #-24]\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "ldr x6, [%[lhs_ptr], #-8]\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "ldr x9, [%[rhs_ptr], #-24]\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"
        "ldr d6, [%[rhs_ptr], #-32]\n"
        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "ldr d0, [%[lhs_ptr], #-64]\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "ldr d1, [%[lhs_ptr], #-48]\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "ins v4.d[1], x7\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ins v5.d[1], x8\n"

        "ldr d2, [%[lhs_ptr], #-32]\n"
        "ins v0.d[1], x3\n"
        "sadalp  v24.4s, v8.8h\n"
        "ldr d3, [%[lhs_ptr], #-16]\n"
        "ins v1.d[1], x4\n"
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "ins v2.d[1], x5\n"
        "sadalp  v25.4s, v9.8h\n"
        "ins v3.d[1], x6\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "ldr d7, [%[rhs_ptr], #-16]\n"
        "sadalp  v26.4s, v10.8h\n"
        "ldr x10, [%[rhs_ptr], #-8]\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "sadalp  v27.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "sadalp  v28.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "sadalp  v29.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "sadalp  v30.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "ins v6.d[1], x9\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "ins v7.d[1], x10\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        "blt 2b\n"

        "79:\n"

        "sadalp  v16.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "sadalp  v25.4s, v9.8h\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "sadalp  v26.4s, v10.8h\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "sadalp  v27.4s, v11.8h\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "sadalp  v28.4s, v12.8h\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "sadalp  v29.4s, v13.8h\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 4x4 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x4 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v17.4s\n"
        "addp v18.4s, v18.4s, v19.4s\n"
        "addp v20.4s, v20.4s, v21.4s\n"
        "addp v22.4s, v22.4s, v23.4s\n"
        "addp v24.4s, v24.4s, v25.4s\n"
        "addp v26.4s, v26.4s, v27.4s\n"
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v16.4s, v16.4s, v18.4s\n"
        "addp v17.4s, v20.4s, v22.4s\n"
        "addp v18.4s, v24.4s, v26.4s\n"
        "addp v19.4s, v28.4s, v30.4s\n"

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 4 bias values.
        "ld1 {v14.4s}, [x1]\n"

        // Load the multiplier_fixedpoint values.
        "add x5, x4, x3, lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"
        "ld1 {v15.4s}, [x4]\n" // multiplier_fixedpoint

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "ldr d0, [%[lhs_ptr], #0]\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows

        "add v16.4s, v16.4s, v14.4s\n"
        "ldr d1, [%[lhs_ptr], #16]\n"
        "add v17.4s, v17.4s, v14.4s\n"
        "ldr d2, [%[lhs_ptr], #32]\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "ldr d3, [%[lhs_ptr], #48]\n"
        "add v19.4s, v19.4s, v14.4s\n"
        "ldr d4, [%[rhs_ptr], #0]\n"
        "ldr d5, [%[rhs_ptr], #16]\n"
        "ldr d6, [%[rhs_ptr], #32]\n"
        "ldr d7, [%[rhs_ptr], #48]\n"

        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v20.4s, v14.s[0]\n"
        "ldr d1, [%[lhs_ptr], #16]\n"
        "dup v21.4s, v14.s[1]\n"
        "ldr d2, [%[lhs_ptr], #32]\n"
        "dup v22.4s, v14.s[2]\n"
        "ldr d3, [%[lhs_ptr], #48]\n"
        "dup v23.4s, v14.s[3]\n"
        "ldr d4, [%[rhs_ptr], #0]\n"
        "add v16.4s, v16.4s, v20.4s\n"
        "ldr d5, [%[rhs_ptr], #16]\n"
        "add v17.4s, v17.4s, v21.4s\n"
        "ldr d6, [%[rhs_ptr], #32]\n"
        "add v18.4s, v18.4s, v22.4s\n"
        "ldr d7, [%[rhs_ptr], #48]\n"
        "add v19.4s, v19.4s, v23.4s\n"
        "7:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[1]\n"
        "mls v18.4s, v10.4s, v14.s[2]\n"
        "mls v19.4s, v10.4s, v14.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v11.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v11.4s\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, x3, lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ld1 {v14.4s}, [x1]\n"

        "smin v11.4s, v8.4s, v14.4s\n"
        "ldr x1, [%[lhs_ptr], #8]\n"
        "sub v12.4s, v14.4s, v11.4s\n"

        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 8f\n"
        // Case where channels are rows


        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v12.4s\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "ldr x3, [%[lhs_ptr], #40]\n"
        "sshl v18.4s, v18.4s, v12.4s\n"
        "ldr x4, [%[lhs_ptr], #56]\n"
        "sshl v19.4s, v19.4s, v12.4s\n"


        // Apply the fixed-point part of the multiplier.
        "ins v0.d[1], x1\n"
        "ldr x1, [%[rhs_ptr], #8]\n"
        "sqdmulh v16.4s, v16.4s, v15.4s\n"
        "ins v1.d[1], x2\n"
        "ldr x2, [%[rhs_ptr], #24]\n"
        "sqdmulh v17.4s, v17.4s, v15.4s\n"
        "ins v2.d[1], x3\n"
        "ldr x3, [%[rhs_ptr], #40]\n"
        "sqdmulh v18.4s, v18.4s, v15.4s\n"
        "ins v3.d[1], x4\n"
        "ldr x4, [%[rhs_ptr], #56]\n"
        "sqdmulh v19.4s, v19.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v11.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v11.4s\n"

        "b 9f\n"

        "8:\n"
        // Case where channels are columns

        // Apply the positive exponent part of the multiplier.
        "dup v20.4s, v12.s[0]\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        "ldr x3, [%[lhs_ptr], #40]\n"
        "dup v21.4s, v12.s[1]\n"
        "ldr x4, [%[lhs_ptr], #56]\n"
        "dup v22.4s, v12.s[2]\n"
        "ins v0.d[1], x1\n"
        "dup v23.4s, v12.s[3]\n"
        "ldr x1, [%[rhs_ptr], #8]\n"
        "sshl v16.4s, v16.4s, v20.4s\n"
        "ins v1.d[1], x2\n"
        "sshl v17.4s, v17.4s, v21.4s\n"
        "ldr x2, [%[rhs_ptr], #24]\n"
        "sshl v18.4s, v18.4s, v22.4s\n"
        "ins v2.d[1], x3\n"
        "sshl v19.4s, v19.4s, v23.4s\n"
        "ldr x3, [%[rhs_ptr], #40]\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v15.s[0]\n"
        "ins v3.d[1], x4\n"
        "sqdmulh v17.4s, v17.4s, v15.s[1]\n"
        "ldr x4, [%[rhs_ptr], #56]\n"
        "sqdmulh v18.4s, v18.4s, v15.s[2]\n"
        "dup v20.4s, v11.s[0]\n"
        "sqdmulh v19.4s, v19.4s, v15.s[3]\n"

        // Apply the negative exponent part of the multiplier.
        "dup v21.4s, v11.s[1]\n"
        "srshl v16.4s, v16.4s, v20.4s\n"
        "dup v22.4s, v11.s[2]\n"
        "srshl v17.4s, v17.4s, v21.4s\n"
        "dup v23.4s, v11.s[3]\n"
        "srshl v18.4s, v18.4s, v22.4s\n"
        "srshl v19.4s, v19.4s, v23.4s\n"

        "9:\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        "ins v4.d[1], x1\n"
        "sqxtn v16.4h, v16.4s\n"
        "ins v5.d[1], x2\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "ins v6.d[1], x3\n"
        "sqxtn v17.4h, v18.4s\n"
        "ins v7.d[1], x4\n"
        RUY_MAKE_ZERO(v18)
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "dup v14.8h, v13.h[4]\n"
        RUY_MAKE_ZERO(v20)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        RUY_MAKE_ZERO(v21)
        "sqadd v17.8h, v17.8h, v14.8h\n"
        RUY_MAKE_ZERO(v22)

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        RUY_MAKE_ZERO(v23)
        "sqxtun2 v16.16b, v17.8h\n"
        RUY_MAKE_ZERO(v24)

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        RUY_MAKE_ZERO(v25)
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        RUY_MAKE_ZERO(v26)
        "dup v14.16b, w2\n"  // clamp_min
        RUY_MAKE_ZERO(v27)
        "dup v15.16b, w3\n"  // clamp_max
        RUY_MAKE_ZERO(v28)

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        RUY_MAKE_ZERO(v29)
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        "ins v4.d[1], x1\n"
        "sqxtn v16.4h, v16.4s\n"
        "ins v5.d[1], x2\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "ins v6.d[1], x3\n"
        "sqxtn v17.4h, v18.4s\n"
        "ins v7.d[1], x4\n"
        RUY_MAKE_ZERO(v18)
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "dup v14.8h, v13.h[4]\n"
        RUY_MAKE_ZERO(v20)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        RUY_MAKE_ZERO(v21)
        "sqadd v17.8h, v17.8h, v14.8h\n"
        RUY_MAKE_ZERO(v22)

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        RUY_MAKE_ZERO(v23)
        "sqxtn2 v16.16b, v17.8h\n"
        RUY_MAKE_ZERO(v24)

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        RUY_MAKE_ZERO(v25)
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        RUY_MAKE_ZERO(v26)
        "dup v14.16b, w2\n"  // clamp_min
        RUY_MAKE_ZERO(v27)
        "dup v15.16b, w3\n"  // clamp_max
        RUY_MAKE_ZERO(v28)

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        RUY_MAKE_ZERO(v29)
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.4h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "ins v4.d[1], x1\n"
        "sqxtn v16.4h, v16.4s\n"
        "ins v5.d[1], x2\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "ins v6.d[1], x3\n"
        "sqxtn v17.4h, v18.4s\n"
        "ins v7.d[1], x4\n"
        RUY_MAKE_ZERO(v18)
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v19)

        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        RUY_MAKE_ZERO(v20)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)

        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)

        // Load the clamp_min, clamp_max bounds
        "ldrh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        RUY_MAKE_ZERO(v25)
        "ldrh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        RUY_MAKE_ZERO(v26)
        "dup v14.8h, w2\n"  // clamp_min
        RUY_MAKE_ZERO(v27)
        "dup v15.8h, w3\n"  // clamp_max
        RUY_MAKE_ZERO(v28)

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        RUY_MAKE_ZERO(v29)
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[1], [x3], #2\n"
        "st1 {v16.h}[2], [x3], #2\n"
        "st1 {v16.h}[3], [x3], #2\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[5], [x3], #2\n"
        "st1 {v16.h}[6], [x3], #2\n"
        "st1 {v16.h}[7], [x3], #2\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v17.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[1], [x3], #2\n"
        "st1 {v17.h}[2], [x3], #2\n"
        "st1 {v17.h}[3], [x3], #2\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v17.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[5], [x3], #2\n"
        "st1 {v17.h}[6], [x3], #2\n"
        "st1 {v17.h}[7], [x3], #2\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #8\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        "ldr x1, [%[lhs_ptr], #8]\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        "ldr x3, [%[lhs_ptr], #40]\n"
        "ldr x4, [%[lhs_ptr], #56]\n"

        "ins v0.d[1], x1\n"
        "ldr x1, [%[rhs_ptr], #8]\n"
        "ins v1.d[1], x2\n"
        "ldr x2, [%[rhs_ptr], #24]\n"
        "ins v2.d[1], x3\n"
        "ldr x3, [%[rhs_ptr], #40]\n"
        "ins v3.d[1], x4\n"
        "ldr x4, [%[rhs_ptr], #56]\n"
        "ins v4.d[1], x1\n"
        "ins v5.d[1], x2\n"
        "ins v6.d[1], x3\n"
        "ins v7.d[1], x4\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // At this point, v20 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).

        RUY_MAKE_ZERO(v20)
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        RUY_MAKE_ZERO(v21)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        RUY_MAKE_ZERO(v22)

        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        "str q18, [%[dst_tmp_buf], #32]\n"
        "str q19, [%[dst_tmp_buf], #48]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v16.s}[1], [x3], #4\n"
        "st1 {v16.s}[2], [x3], #4\n"
        "st1 {v16.s}[3], [x3], #4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v17.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v17.s}[1], [x3], #4\n"
        "st1 {v17.s}[2], [x3], #4\n"
        "st1 {v17.s}[3], [x3], #4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v18.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v18.s}[1], [x3], #4\n"
        "st1 {v18.s}[2], [x3], #4\n"
        "st1 {v18.s}[3], [x3], #4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v19.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v19.s}[1], [x3], #4\n"
        "st1 {v19.s}[2], [x3], #4\n"
        "st1 {v19.s}[3], [x3], #4\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #16\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #4\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #4\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #16\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params),[dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Kernel taking advantage of the optional dotprod instruction.
// This is very similar to (and directly inspired by) this gemmlowp kernel
// which was contributed by David Mansell at ARM:
// NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_dotproduct
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L3391
//
// Besides the ruy-ification, the main difference here is that we use a 8x8
// instead of 12x8 width, so as to stick to power-of-two widths. This slightly
// narrower kernel layout is still wide enough to achieve high performance
// although we haven't actually performed a real comparison to know exactly
// how this compares to ARM's aforementioned kernel.
//
// Relevant target CPUs for this kernel include ARM Cortex-A76,
// since these are 64-bit, out-of-order and with dotprod support.
void Kernel8bitNeonDotprod(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprod)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v15 are used to load int8 data from LHS and
  // RHS. At least v0 and v1 are used to load a 8x4 block of LHS, and v2 and
  // v3 are used to load a 4x8 block of RHS, like this:
  //
  //                                      int8 RHS 4x8 block
  //                           /-----------------------------------------|
  //                           |v2.b[0] ... v2.b[12] v3.b[0] ... v3.b[12]|
  //                           |  ...                              ...   |
  //                           |v2.b[3] ... v2.b[15] v3.b[3] ... v3.b[15]|
  //                           \-----------------------------------------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /-----------------------------------------|
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]           ...           v30.s[0]|
  //  |  ...          ...   |  |  ...                              ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]           ...           v30.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]           ...           v31.s[0]|
  //  |  ...         ...    |  |  ...                              ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for loading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Optional, maximally-streaming, partial-unrolling (4x unrolled)
        // optimization of the kernel inner loop (over depth). For more
        // comments, see the non-unrolled loop below after the #endif.
#if RUY_OPT(MAX_STREAMING)
        "cmp w12, #32\n"
        "blt 78f\n"

        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v8.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v9.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v10.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v11.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v12.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v13.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v14.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v15.16b}, [%[rhs_ptr]], #16\n"
        "mov w1, #16\n"

        "and w3, w12, #-16\n"
        "81:\n"
        "add w1, w1, #16\n"

        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ldr q0, [%[lhs_ptr], #0]\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ldr q2, [%[rhs_ptr], #0]\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ldr q1, [%[lhs_ptr], #16]\n"

        ".inst 0x4f87e098  // sdot v24.4s, v4.16b, v7.4b[0]\n"
        ".inst 0x4fa7e09a  // sdot v26.4s, v4.16b, v7.4b[1]\n"
        "ldr q3, [%[rhs_ptr], #16]\n"
        ".inst 0x4f87e89c  // sdot v28.4s, v4.16b, v7.4b[2]\n"
        ".inst 0x4fa7e89e  // sdot v30.4s, v4.16b, v7.4b[3]\n"
        ".inst 0x4f86e0b1  // sdot v17.4s, v5.16b, v6.4b[0]\n"
        ".inst 0x4fa6e0b3  // sdot v19.4s, v5.16b, v6.4b[1]\n"
        ".inst 0x4f86e8b5  // sdot v21.4s, v5.16b, v6.4b[2]\n"
        ".inst 0x4fa6e8b7  // sdot v23.4s, v5.16b, v6.4b[3]\n"
        ".inst 0x4f87e0b9  // sdot v25.4s, v5.16b, v7.4b[0]\n"
        ".inst 0x4fa7e0bb  // sdot v27.4s, v5.16b, v7.4b[1]\n"
        ".inst 0x4f87e8bd  // sdot v29.4s, v5.16b, v7.4b[2]\n"
        ".inst 0x4fa7e8bf  // sdot v31.4s, v5.16b, v7.4b[3]\n"
        "ldr q5, [%[lhs_ptr], #48]\n"
        ".inst 0x4f86e090  // sdot v16.4s, v4.16b, v6.4b[0]\n"
        ".inst 0x4fa6e092  // sdot v18.4s, v4.16b, v6.4b[1]\n"
        "ldr q7, [%[rhs_ptr], #48]\n"
        ".inst 0x4f86e894  // sdot v20.4s, v4.16b, v6.4b[2]\n"
        ".inst 0x4fa6e896  // sdot v22.4s, v4.16b, v6.4b[3]\n"
        "ldr q4, [%[lhs_ptr], #32]\n"

        ".inst 0x4f8be118  // sdot v24.4s, v8.16b, v11.4b[0]\n"
        ".inst 0x4fabe11a  // sdot v26.4s, v8.16b, v11.4b[1]\n"
        "ldr q6, [%[rhs_ptr], #32]\n"
        ".inst 0x4f8be91c  // sdot v28.4s, v8.16b, v11.4b[2]\n"
        ".inst 0x4fabe91e  // sdot v30.4s, v8.16b, v11.4b[3]\n"
        ".inst 0x4f8ae131  // sdot v17.4s, v9.16b, v10.4b[0]\n"
        ".inst 0x4faae133  // sdot v19.4s, v9.16b, v10.4b[1]\n"
        ".inst 0x4f8ae935  // sdot v21.4s, v9.16b, v10.4b[2]\n"
        ".inst 0x4faae937  // sdot v23.4s, v9.16b, v10.4b[3]\n"
        ".inst 0x4f8be139  // sdot v25.4s, v9.16b, v11.4b[0]\n"
        ".inst 0x4fabe13b  // sdot v27.4s, v9.16b, v11.4b[1]\n"
        ".inst 0x4f8be93d  // sdot v29.4s, v9.16b, v11.4b[2]\n"
        ".inst 0x4fabe93f  // sdot v31.4s, v9.16b, v11.4b[3]\n"
        "ldr q9, [%[lhs_ptr], #80]\n"
        ".inst 0x4f8ae110  // sdot v16.4s, v8.16b, v10.4b[0]\n"
        ".inst 0x4faae112  // sdot v18.4s, v8.16b, v10.4b[1]\n"
        "ldr q11, [%[rhs_ptr], #80]\n"
        ".inst 0x4f8ae914  // sdot v20.4s, v8.16b, v10.4b[2]\n"
        ".inst 0x4faae916  // sdot v22.4s, v8.16b, v10.4b[3]\n"
        "ldr q8, [%[lhs_ptr], #64]\n"

        ".inst 0x4f8fe198  // sdot v24.4s, v12.16b, v15.4b[0]\n"
        ".inst 0x4fafe19a  // sdot v26.4s, v12.16b, v15.4b[1]\n"
        "ldr q10, [%[rhs_ptr], #64]\n"
        ".inst 0x4f8fe99c  // sdot v28.4s, v12.16b, v15.4b[2]\n"
        ".inst 0x4fafe99e  // sdot v30.4s, v12.16b, v15.4b[3]\n"
        "add %[lhs_ptr], %[lhs_ptr], #128\n"
        ".inst 0x4f8ee1b1  // sdot v17.4s, v13.16b, v14.4b[0]\n"
        ".inst 0x4faee1b3  // sdot v19.4s, v13.16b, v14.4b[1]\n"
        "add %[rhs_ptr], %[rhs_ptr], #128\n"
        ".inst 0x4f8ee9b5  // sdot v21.4s, v13.16b, v14.4b[2]\n"
        ".inst 0x4faee9b7  // sdot v23.4s, v13.16b, v14.4b[3]\n"
        ".inst 0x4f8fe1b9  // sdot v25.4s, v13.16b, v15.4b[0]\n"
        ".inst 0x4fafe1bb  // sdot v27.4s, v13.16b, v15.4b[1]\n"
        "cmp w1, w3\n"
        ".inst 0x4f8fe9bd  // sdot v29.4s, v13.16b, v15.4b[2]\n"
        ".inst 0x4fafe9bf  // sdot v31.4s, v13.16b, v15.4b[3]\n"
        "ldr q13, [%[lhs_ptr], #-16]\n"
        ".inst 0x4f8ee190  // sdot v16.4s, v12.16b, v14.4b[0]\n"
        ".inst 0x4faee192  // sdot v18.4s, v12.16b, v14.4b[1]\n"
        "ldr q15, [%[rhs_ptr], #-16]\n"
        ".inst 0x4f8ee994  // sdot v20.4s, v12.16b, v14.4b[2]\n"
        ".inst 0x4faee996  // sdot v22.4s, v12.16b, v14.4b[3]\n"
        "ldr q12, [%[lhs_ptr], #-32]\n"

        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        "ldr q14, [%[rhs_ptr], #-32]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        "blt 81b\n"

        ".inst 0x4f87e098  // sdot v24.4s, v4.16b, v7.4b[0]\n"
        ".inst 0x4fa7e09a  // sdot v26.4s, v4.16b, v7.4b[1]\n"
        ".inst 0x4f87e89c  // sdot v28.4s, v4.16b, v7.4b[2]\n"
        ".inst 0x4fa7e89e  // sdot v30.4s, v4.16b, v7.4b[3]\n"
        ".inst 0x4f86e0b1  // sdot v17.4s, v5.16b, v6.4b[0]\n"
        ".inst 0x4fa6e0b3  // sdot v19.4s, v5.16b, v6.4b[1]\n"
        ".inst 0x4f86e8b5  // sdot v21.4s, v5.16b, v6.4b[2]\n"
        ".inst 0x4fa6e8b7  // sdot v23.4s, v5.16b, v6.4b[3]\n"
        ".inst 0x4f87e0b9  // sdot v25.4s, v5.16b, v7.4b[0]\n"
        ".inst 0x4fa7e0bb  // sdot v27.4s, v5.16b, v7.4b[1]\n"
        ".inst 0x4f87e8bd  // sdot v29.4s, v5.16b, v7.4b[2]\n"
        ".inst 0x4fa7e8bf  // sdot v31.4s, v5.16b, v7.4b[3]\n"
        ".inst 0x4f86e090  // sdot v16.4s, v4.16b, v6.4b[0]\n"
        ".inst 0x4fa6e092  // sdot v18.4s, v4.16b, v6.4b[1]\n"
        ".inst 0x4f86e894  // sdot v20.4s, v4.16b, v6.4b[2]\n"
        ".inst 0x4fa6e896  // sdot v22.4s, v4.16b, v6.4b[3]\n"

        ".inst 0x4f8be118  // sdot v24.4s, v8.16b, v11.4b[0]\n"
        ".inst 0x4fabe11a  // sdot v26.4s, v8.16b, v11.4b[1]\n"
        ".inst 0x4f8be91c  // sdot v28.4s, v8.16b, v11.4b[2]\n"
        ".inst 0x4fabe91e  // sdot v30.4s, v8.16b, v11.4b[3]\n"
        ".inst 0x4f8ae131  // sdot v17.4s, v9.16b, v10.4b[0]\n"
        ".inst 0x4faae133  // sdot v19.4s, v9.16b, v10.4b[1]\n"
        ".inst 0x4f8ae935  // sdot v21.4s, v9.16b, v10.4b[2]\n"
        ".inst 0x4faae937  // sdot v23.4s, v9.16b, v10.4b[3]\n"
        ".inst 0x4f8be139  // sdot v25.4s, v9.16b, v11.4b[0]\n"
        ".inst 0x4fabe13b  // sdot v27.4s, v9.16b, v11.4b[1]\n"
        ".inst 0x4f8be93d  // sdot v29.4s, v9.16b, v11.4b[2]\n"
        ".inst 0x4fabe93f  // sdot v31.4s, v9.16b, v11.4b[3]\n"
        ".inst 0x4f8ae110  // sdot v16.4s, v8.16b, v10.4b[0]\n"
        ".inst 0x4faae112  // sdot v18.4s, v8.16b, v10.4b[1]\n"
        ".inst 0x4f8ae914  // sdot v20.4s, v8.16b, v10.4b[2]\n"
        ".inst 0x4faae916  // sdot v22.4s, v8.16b, v10.4b[3]\n"

        ".inst 0x4f8fe198  // sdot v24.4s, v12.16b, v15.4b[0]\n"
        ".inst 0x4fafe19a  // sdot v26.4s, v12.16b, v15.4b[1]\n"
        ".inst 0x4f8fe99c  // sdot v28.4s, v12.16b, v15.4b[2]\n"
        ".inst 0x4fafe99e  // sdot v30.4s, v12.16b, v15.4b[3]\n"
        ".inst 0x4f8ee1b1  // sdot v17.4s, v13.16b, v14.4b[0]\n"
        ".inst 0x4faee1b3  // sdot v19.4s, v13.16b, v14.4b[1]\n"
        ".inst 0x4f8ee9b5  // sdot v21.4s, v13.16b, v14.4b[2]\n"
        ".inst 0x4faee9b7  // sdot v23.4s, v13.16b, v14.4b[3]\n"
        ".inst 0x4f8fe1b9  // sdot v25.4s, v13.16b, v15.4b[0]\n"
        ".inst 0x4fafe1bb  // sdot v27.4s, v13.16b, v15.4b[1]\n"
        ".inst 0x4f8fe9bd  // sdot v29.4s, v13.16b, v15.4b[2]\n"
        ".inst 0x4fafe9bf  // sdot v31.4s, v13.16b, v15.4b[3]\n"
        ".inst 0x4f8ee190  // sdot v16.4s, v12.16b, v14.4b[0]\n"
        ".inst 0x4faee192  // sdot v18.4s, v12.16b, v14.4b[1]\n"
        ".inst 0x4f8ee994  // sdot v20.4s, v12.16b, v14.4b[2]\n"
        ".inst 0x4faee996  // sdot v22.4s, v12.16b, v14.4b[3]\n"

        "78:\n"

#endif  // #if RUY_OPT(MAX_STREAMING)

        // Ordinary kernel inner loop (over depth), the simpler loop that the
        // above was an equivalent 4x-partially-unrolled version of.

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Because of the data that we have already loaded, we can start the
        // loop body right away with some multiply-adds.
        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        // Each iteration of this loop advances by 4 levels of depth.
        "add w1, w1, #4\n"
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        // Loop termination condition.
        "cmp w1, w12\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

        "blt 2b\n"

        "79:\n"
        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last 4 levels of depth, for which the LHS
        // and RHS data is already loaded.

        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v15.4s\n"
        "add v20.4s, v20.4s, v14.4s\n"
        "add v21.4s, v21.4s, v15.4s\n"
        "add v22.4s, v22.4s, v14.4s\n"
        "add v23.4s, v23.4s, v15.4s\n"
        "add v24.4s, v24.4s, v14.4s\n"
        "add v25.4s, v25.4s, v15.4s\n"
        "add v26.4s, v26.4s, v14.4s\n"
        "add v27.4s, v27.4s, v15.4s\n"
        "add v28.4s, v28.4s, v14.4s\n"
        "add v29.4s, v29.4s, v15.4s\n"
        "add v30.4s, v30.4s, v14.4s\n"
        "add v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v10.4s, v14.s[0]\n"
        "dup v11.4s, v14.s[1]\n"
        "dup v12.4s, v14.s[2]\n"
        "dup v13.4s, v14.s[3]\n"
        "add v16.4s, v16.4s, v10.4s\n"
        "add v17.4s, v17.4s, v10.4s\n"
        "add v18.4s, v18.4s, v11.4s\n"
        "add v19.4s, v19.4s, v11.4s\n"
        "add v20.4s, v20.4s, v12.4s\n"
        "add v21.4s, v21.4s, v12.4s\n"
        "add v22.4s, v22.4s, v13.4s\n"
        "add v23.4s, v23.4s, v13.4s\n"
        "dup v10.4s, v15.s[0]\n"
        "dup v11.4s, v15.s[1]\n"
        "dup v12.4s, v15.s[2]\n"
        "dup v13.4s, v15.s[3]\n"
        "add v24.4s, v24.4s, v10.4s\n"
        "add v25.4s, v25.4s, v10.4s\n"
        "add v26.4s, v26.4s, v11.4s\n"
        "add v27.4s, v27.4s, v11.4s\n"
        "add v28.4s, v28.4s, v12.4s\n"
        "add v29.4s, v29.4s, v12.4s\n"
        "add v30.4s, v30.4s, v13.4s\n"
        "add v31.4s, v31.4s, v13.4s\n"
        "7:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3], #16\n"
        "ld1 {v15.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "mls v18.4s, v10.4s, v14.s[1]\n"
        "mls v19.4s, v10.4s, v14.s[1]\n"
        "mls v20.4s, v10.4s, v14.s[2]\n"
        "mls v21.4s, v10.4s, v14.s[2]\n"
        "mls v22.4s, v10.4s, v14.s[3]\n"
        "mls v23.4s, v10.4s, v14.s[3]\n"
        "mls v24.4s, v10.4s, v15.s[0]\n"
        "mls v25.4s, v10.4s, v15.s[0]\n"
        "mls v26.4s, v10.4s, v15.s[1]\n"
        "mls v27.4s, v10.4s, v15.s[1]\n"
        "mls v28.4s, v10.4s, v15.s[2]\n"
        "mls v29.4s, v10.4s, v15.s[2]\n"
        "mls v30.4s, v10.4s, v15.s[3]\n"
        "mls v31.4s, v10.4s, v15.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2], #16\n"
        "ld1 {v12.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v12.4s\n"
        "sub v20.4s, v20.4s, v11.4s\n"
        "sub v21.4s, v21.4s, v12.4s\n"
        "sub v22.4s, v22.4s, v11.4s\n"
        "sub v23.4s, v23.4s, v12.4s\n"
        "sub v24.4s, v24.4s, v11.4s\n"
        "sub v25.4s, v25.4s, v12.4s\n"
        "sub v26.4s, v26.4s, v11.4s\n"
        "sub v27.4s, v27.4s, v12.4s\n"
        "sub v28.4s, v28.4s, v11.4s\n"
        "sub v29.4s, v29.4s, v12.4s\n"
        "sub v30.4s, v30.4s, v11.4s\n"
        "sub v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"
        // Compute the multiplier_exponent pointer
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, x3, lsl #2\n"
        "csel x1, x1, x5, eq\n"
        // Load multiplier_exponent
        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"
        // Separate positive and negative exponents
        "smin v11.4s, v8.4s, v9.4s\n"
        "smin v12.4s, v8.4s, v10.4s\n"
        "sub v9.4s, v9.4s, v11.4s\n"
        "sub v10.4s, v10.4s, v12.4s\n"

        // Compute the multiplier_fixedpoint pointer
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "add x5, x4, x3, lsl #2\n"
        "csel x4, x4, x5, eq\n"
        // Load multiplier_fixedpoint
        "ldr q14, [x4]\n"
        "ldr q15, [x4, #16]\n"

        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 8f\n"
        // Case where channels are rows

        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v9.4s\n"
        "sshl v17.4s, v17.4s, v10.4s\n"
        "sshl v18.4s, v18.4s, v9.4s\n"
        "sshl v19.4s, v19.4s, v10.4s\n"
        "sshl v20.4s, v20.4s, v9.4s\n"
        "sshl v21.4s, v21.4s, v10.4s\n"
        "sshl v22.4s, v22.4s, v9.4s\n"
        "sshl v23.4s, v23.4s, v10.4s\n"
        "sshl v24.4s, v24.4s, v9.4s\n"
        "sshl v25.4s, v25.4s, v10.4s\n"
        "sshl v26.4s, v26.4s, v9.4s\n"
        "sshl v27.4s, v27.4s, v10.4s\n"
        "sshl v28.4s, v28.4s, v9.4s\n"
        "sshl v29.4s, v29.4s, v10.4s\n"
        "sshl v30.4s, v30.4s, v9.4s\n"
        "sshl v31.4s, v31.4s, v10.4s\n"
        "10:\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v14.4s\n"
        "sqdmulh v17.4s, v17.4s, v15.4s\n"
        "sqdmulh v18.4s, v18.4s, v14.4s\n"
        "sqdmulh v19.4s, v19.4s, v15.4s\n"
        "sqdmulh v20.4s, v20.4s, v14.4s\n"
        "sqdmulh v21.4s, v21.4s, v15.4s\n"
        "sqdmulh v22.4s, v22.4s, v14.4s\n"
        "sqdmulh v23.4s, v23.4s, v15.4s\n"
        "sqdmulh v24.4s, v24.4s, v14.4s\n"
        "sqdmulh v25.4s, v25.4s, v15.4s\n"
        "sqdmulh v26.4s, v26.4s, v14.4s\n"
        "sqdmulh v27.4s, v27.4s, v15.4s\n"
        "sqdmulh v28.4s, v28.4s, v14.4s\n"
        "sqdmulh v29.4s, v29.4s, v15.4s\n"
        "sqdmulh v30.4s, v30.4s, v14.4s\n"
        "sqdmulh v31.4s, v31.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"
        "srshl v20.4s, v20.4s, v11.4s\n"
        "srshl v21.4s, v21.4s, v12.4s\n"
        "srshl v22.4s, v22.4s, v11.4s\n"
        "srshl v23.4s, v23.4s, v12.4s\n"
        "srshl v24.4s, v24.4s, v11.4s\n"
        "srshl v25.4s, v25.4s, v12.4s\n"
        "srshl v26.4s, v26.4s, v11.4s\n"
        "srshl v27.4s, v27.4s, v12.4s\n"
        "srshl v28.4s, v28.4s, v11.4s\n"
        "srshl v29.4s, v29.4s, v12.4s\n"
        "srshl v30.4s, v30.4s, v11.4s\n"
        "srshl v31.4s, v31.4s, v12.4s\n"
        "b 9f\n"

        "8:\n"
        // Case where channels are columns

        // Apply the positive exponent part of the multiplier.
        "dup v4.4s, v9.s[0]\n"
        "dup v5.4s, v9.s[1]\n"
        "dup v6.4s, v9.s[2]\n"
        "dup v7.4s, v9.s[3]\n"
        "sshl v16.4s, v16.4s, v4.4s\n"
        "sshl v17.4s, v17.4s, v4.4s\n"
        "sshl v18.4s, v18.4s, v5.4s\n"
        "sshl v19.4s, v19.4s, v5.4s\n"
        "sshl v20.4s, v20.4s, v6.4s\n"
        "sshl v21.4s, v21.4s, v6.4s\n"
        "sshl v22.4s, v22.4s, v7.4s\n"
        "sshl v23.4s, v23.4s, v7.4s\n"
        "dup v4.4s, v10.s[0]\n"
        "dup v5.4s, v10.s[1]\n"
        "dup v6.4s, v10.s[2]\n"
        "dup v7.4s, v10.s[3]\n"
        "sshl v24.4s, v24.4s, v4.4s\n"
        "sshl v25.4s, v25.4s, v4.4s\n"
        "sshl v26.4s, v26.4s, v5.4s\n"
        "sshl v27.4s, v27.4s, v5.4s\n"
        "sshl v28.4s, v28.4s, v6.4s\n"
        "sshl v29.4s, v29.4s, v6.4s\n"
        "sshl v30.4s, v30.4s, v7.4s\n"
        "sshl v31.4s, v31.4s, v7.4s\n"
        "11:\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v14.s[0]\n"
        "sqdmulh v17.4s, v17.4s, v14.s[0]\n"
        "sqdmulh v18.4s, v18.4s, v14.s[1]\n"
        "sqdmulh v19.4s, v19.4s, v14.s[1]\n"
        "sqdmulh v20.4s, v20.4s, v14.s[2]\n"
        "sqdmulh v21.4s, v21.4s, v14.s[2]\n"
        "sqdmulh v22.4s, v22.4s, v14.s[3]\n"
        "sqdmulh v23.4s, v23.4s, v14.s[3]\n"
        "sqdmulh v24.4s, v24.4s, v15.s[0]\n"
        "sqdmulh v25.4s, v25.4s, v15.s[0]\n"
        "sqdmulh v26.4s, v26.4s, v15.s[1]\n"
        "sqdmulh v27.4s, v27.4s, v15.s[1]\n"
        "sqdmulh v28.4s, v28.4s, v15.s[2]\n"
        "sqdmulh v29.4s, v29.4s, v15.s[2]\n"
        "sqdmulh v30.4s, v30.4s, v15.s[3]\n"
        "sqdmulh v31.4s, v31.4s, v15.s[3]\n"

        // Apply the negative exponent part of the multiplier.
        "dup v4.4s, v11.s[0]\n"
        "dup v5.4s, v11.s[1]\n"
        "dup v6.4s, v11.s[2]\n"
        "dup v7.4s, v11.s[3]\n"
        "srshl v16.4s, v16.4s, v4.4s\n"
        "srshl v17.4s, v17.4s, v4.4s\n"
        "srshl v18.4s, v18.4s, v5.4s\n"
        "srshl v19.4s, v19.4s, v5.4s\n"
        "srshl v20.4s, v20.4s, v6.4s\n"
        "srshl v21.4s, v21.4s, v6.4s\n"
        "srshl v22.4s, v22.4s, v7.4s\n"
        "srshl v23.4s, v23.4s, v7.4s\n"
        "dup v4.4s, v12.s[0]\n"
        "dup v5.4s, v12.s[1]\n"
        "dup v6.4s, v12.s[2]\n"
        "dup v7.4s, v12.s[3]\n"
        "srshl v24.4s, v24.4s, v4.4s\n"
        "srshl v25.4s, v25.4s, v4.4s\n"
        "srshl v26.4s, v26.4s, v5.4s\n"
        "srshl v27.4s, v27.4s, v5.4s\n"
        "srshl v28.4s, v28.4s, v6.4s\n"
        "srshl v29.4s, v29.4s, v6.4s\n"
        "srshl v30.4s, v30.4s, v7.4s\n"
        "srshl v31.4s, v31.4s, v7.4s\n"
        "9:\n"

        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"
        "sqadd v18.8h, v18.8h, v14.8h\n"
        "sqadd v19.8h, v19.8h, v14.8h\n"
        "sqadd v20.8h, v20.8h, v14.8h\n"
        "sqadd v21.8h, v21.8h, v14.8h\n"
        "sqadd v22.8h, v22.8h, v14.8h\n"
        "sqadd v23.8h, v23.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "sqxtun2 v16.16b, v17.8h\n"
        "sqxtun v17.8b, v18.8h\n"
        "sqxtun2 v17.16b, v19.8h\n"
        "sqxtun v18.8b, v20.8h\n"
        "sqxtun2 v18.16b, v21.8h\n"
        "sqxtun v19.8b, v22.8h\n"
        "sqxtun2 v19.16b, v23.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        "umax v17.16b, v17.16b, v14.16b\n"
        "umax v18.16b, v18.16b, v14.16b\n"
        "umax v19.16b, v19.16b, v14.16b\n"

        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        "umin v17.16b, v17.16b, v15.16b\n"
        "umin v18.16b, v18.16b, v15.16b\n"
        "umin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"
        "sqadd v18.8h, v18.8h, v14.8h\n"
        "sqadd v19.8h, v19.8h, v14.8h\n"
        "sqadd v20.8h, v20.8h, v14.8h\n"
        "sqadd v21.8h, v21.8h, v14.8h\n"
        "sqadd v22.8h, v22.8h, v14.8h\n"
        "sqadd v23.8h, v23.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        "sqxtn2 v16.16b, v17.8h\n"
        "sqxtn v17.8b, v18.8h\n"
        "sqxtn2 v17.16b, v19.8h\n"
        "sqxtn v18.8b, v20.8h\n"
        "sqxtn2 v18.16b, v21.8h\n"
        "sqxtn v19.8b, v22.8h\n"
        "sqxtn2 v19.16b, v23.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        "smax v17.16b, v17.16b, v14.16b\n"
        "smax v18.16b, v18.16b, v14.16b\n"
        "smax v19.16b, v19.16b, v14.16b\n"

        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        "smin v17.16b, v17.16b, v15.16b\n"
        "smin v18.16b, v18.16b, v15.16b\n"
        "smin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 150b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"
        "saddw v20.4s, v20.4s, v14.4h\n"
        "saddw v21.4s, v21.4s, v14.4h\n"
        "saddw v22.4s, v22.4s, v14.4h\n"
        "saddw v23.4s, v23.4s, v14.4h\n"
        "saddw v24.4s, v24.4s, v14.4h\n"
        "saddw v25.4s, v25.4s, v14.4h\n"
        "saddw v26.4s, v26.4s, v14.4h\n"
        "saddw v27.4s, v27.4s, v14.4h\n"
        "saddw v28.4s, v28.4s, v14.4h\n"
        "saddw v29.4s, v29.4s, v14.4h\n"
        "saddw v30.4s, v30.4s, v14.4h\n"
        "saddw v31.4s, v31.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        "smax v18.8h, v18.8h, v14.8h\n"
        "smax v19.8h, v19.8h, v14.8h\n"
        "smax v20.8h, v20.8h, v14.8h\n"
        "smax v21.8h, v21.8h, v14.8h\n"
        "smax v22.8h, v22.8h, v14.8h\n"
        "smax v23.8h, v23.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        "smin v18.8h, v18.8h, v15.8h\n"
        "smin v19.8h, v19.8h, v15.8h\n"
        "smin v20.8h, v20.8h, v15.8h\n"
        "smin v21.8h, v21.8h, v15.8h\n"
        "smin v22.8h, v22.8h, v15.8h\n"
        "smin v23.8h, v23.8h, v15.8h\n"

        // Compute how much of the 8x8 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 16bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 250b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x8 block of destination 32it values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x8 block fits.
        // Write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "st1 {v16.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v23)
        "st1 {v24.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v24)
        "st1 {v25.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v25)
        "st1 {v26.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v26)
        "st1 {v27.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v27)
        "st1 {v28.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v28)
        "st1 {v29.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v29)
        "st1 {v30.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v30)
        "st1 {v31.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v31)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x8 block fits.
        "mov x4, %[dst_ptr]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.4s, v17.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v18.4s, v19.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v20.4s, v21.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v22.4s, v23.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v24.4s, v25.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v26.4s, v27.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v28.4s, v29.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v30.4s, v31.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 350b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// A fork of the above 8bitNeonDotprod kernel but removes the max streaming
// manual unrolling. Manually unrolling the inner loops benefits some GEMM
// shapes on the Cortex-A76 but destroys performance on the X1 by increasing
// backend stalls. Therefore, we remove the MAX_STREAMING option in this
// kernel. The target CPU for this kernel is currently only the Cortex-X1.
void Kernel8bitNeonDotprodX1(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprod)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v15 are used to load int8 data from LHS and
  // RHS. At least v0 and v1 are used to load a 8x4 block of LHS, and v2 and
  // v3 are used to load a 4x8 block of RHS, like this:
  //
  //                                      int8 RHS 4x8 block
  //                           /-----------------------------------------|
  //                           |v2.b[0] ... v2.b[12] v3.b[0] ... v3.b[12]|
  //                           |  ...                              ...   |
  //                           |v2.b[3] ... v2.b[15] v3.b[3] ... v3.b[15]|
  //                           \-----------------------------------------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /-----------------------------------------|
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]           ...           v30.s[0]|
  //  |  ...          ...   |  |  ...                              ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]           ...           v30.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]           ...           v31.s[0]|
  //  |  ...         ...    |  |  ...                              ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for loading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Kernel inner loop (over depth).
        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Because of the data that we have already loaded, we can start the
        // loop body right away with some multiply-adds.
        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        // Each iteration of this loop advances by 4 levels of depth.
        "add w1, w1, #4\n"
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        // Loop termination condition.
        "cmp w1, w12\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

        "blt 2b\n"

        "79:\n"
        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last 4 levels of depth, for which the LHS
        // and RHS data is already loaded.

        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v15.4s\n"
        "add v20.4s, v20.4s, v14.4s\n"
        "add v21.4s, v21.4s, v15.4s\n"
        "add v22.4s, v22.4s, v14.4s\n"
        "add v23.4s, v23.4s, v15.4s\n"
        "add v24.4s, v24.4s, v14.4s\n"
        "add v25.4s, v25.4s, v15.4s\n"
        "add v26.4s, v26.4s, v14.4s\n"
        "add v27.4s, v27.4s, v15.4s\n"
        "add v28.4s, v28.4s, v14.4s\n"
        "add v29.4s, v29.4s, v15.4s\n"
        "add v30.4s, v30.4s, v14.4s\n"
        "add v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v10.4s, v14.s[0]\n"
        "dup v11.4s, v14.s[1]\n"
        "dup v12.4s, v14.s[2]\n"
        "dup v13.4s, v14.s[3]\n"
        "add v16.4s, v16.4s, v10.4s\n"
        "add v17.4s, v17.4s, v10.4s\n"
        "add v18.4s, v18.4s, v11.4s\n"
        "add v19.4s, v19.4s, v11.4s\n"
        "add v20.4s, v20.4s, v12.4s\n"
        "add v21.4s, v21.4s, v12.4s\n"
        "add v22.4s, v22.4s, v13.4s\n"
        "add v23.4s, v23.4s, v13.4s\n"
        "dup v10.4s, v15.s[0]\n"
        "dup v11.4s, v15.s[1]\n"
        "dup v12.4s, v15.s[2]\n"
        "dup v13.4s, v15.s[3]\n"
        "add v24.4s, v24.4s, v10.4s\n"
        "add v25.4s, v25.4s, v10.4s\n"
        "add v26.4s, v26.4s, v11.4s\n"
        "add v27.4s, v27.4s, v11.4s\n"
        "add v28.4s, v28.4s, v12.4s\n"
        "add v29.4s, v29.4s, v12.4s\n"
        "add v30.4s, v30.4s, v13.4s\n"
        "add v31.4s, v31.4s, v13.4s\n"
        "7:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3], #16\n"
        "ld1 {v15.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "mls v18.4s, v10.4s, v14.s[1]\n"
        "mls v19.4s, v10.4s, v14.s[1]\n"
        "mls v20.4s, v10.4s, v14.s[2]\n"
        "mls v21.4s, v10.4s, v14.s[2]\n"
        "mls v22.4s, v10.4s, v14.s[3]\n"
        "mls v23.4s, v10.4s, v14.s[3]\n"
        "mls v24.4s, v10.4s, v15.s[0]\n"
        "mls v25.4s, v10.4s, v15.s[0]\n"
        "mls v26.4s, v10.4s, v15.s[1]\n"
        "mls v27.4s, v10.4s, v15.s[1]\n"
        "mls v28.4s, v10.4s, v15.s[2]\n"
        "mls v29.4s, v10.4s, v15.s[2]\n"
        "mls v30.4s, v10.4s, v15.s[3]\n"
        "mls v31.4s, v10.4s, v15.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2], #16\n"
        "ld1 {v12.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v12.4s\n"
        "sub v20.4s, v20.4s, v11.4s\n"
        "sub v21.4s, v21.4s, v12.4s\n"
        "sub v22.4s, v22.4s, v11.4s\n"
        "sub v23.4s, v23.4s, v12.4s\n"
        "sub v24.4s, v24.4s, v11.4s\n"
        "sub v25.4s, v25.4s, v12.4s\n"
        "sub v26.4s, v26.4s, v11.4s\n"
        "sub v27.4s, v27.4s, v12.4s\n"
        "sub v28.4s, v28.4s, v11.4s\n"
        "sub v29.4s, v29.4s, v12.4s\n"
        "sub v30.4s, v30.4s, v11.4s\n"
        "sub v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"
        // Compute the multiplier_exponent pointer
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, x3, lsl #2\n"
        "csel x1, x1, x5, eq\n"
        // Load multiplier_exponent
        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"
        // Separate positive and negative exponents
        "smin v11.4s, v8.4s, v9.4s\n"
        "smin v12.4s, v8.4s, v10.4s\n"
        "sub v9.4s, v9.4s, v11.4s\n"
        "sub v10.4s, v10.4s, v12.4s\n"

        // Compute the multiplier_fixedpoint pointer
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "add x5, x4, x3, lsl #2\n"
        "csel x4, x4, x5, eq\n"
        // Load multiplier_fixedpoint
        "ldr q14, [x4]\n"
        "ldr q15, [x4, #16]\n"

        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 8f\n"
        // Case where channels are rows

        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v9.4s\n"
        "sshl v17.4s, v17.4s, v10.4s\n"
        "sshl v18.4s, v18.4s, v9.4s\n"
        "sshl v19.4s, v19.4s, v10.4s\n"
        "sshl v20.4s, v20.4s, v9.4s\n"
        "sshl v21.4s, v21.4s, v10.4s\n"
        "sshl v22.4s, v22.4s, v9.4s\n"
        "sshl v23.4s, v23.4s, v10.4s\n"
        "sshl v24.4s, v24.4s, v9.4s\n"
        "sshl v25.4s, v25.4s, v10.4s\n"
        "sshl v26.4s, v26.4s, v9.4s\n"
        "sshl v27.4s, v27.4s, v10.4s\n"
        "sshl v28.4s, v28.4s, v9.4s\n"
        "sshl v29.4s, v29.4s, v10.4s\n"
        "sshl v30.4s, v30.4s, v9.4s\n"
        "sshl v31.4s, v31.4s, v10.4s\n"
        "10:\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v14.4s\n"
        "sqdmulh v17.4s, v17.4s, v15.4s\n"
        "sqdmulh v18.4s, v18.4s, v14.4s\n"
        "sqdmulh v19.4s, v19.4s, v15.4s\n"
        "sqdmulh v20.4s, v20.4s, v14.4s\n"
        "sqdmulh v21.4s, v21.4s, v15.4s\n"
        "sqdmulh v22.4s, v22.4s, v14.4s\n"
        "sqdmulh v23.4s, v23.4s, v15.4s\n"
        "sqdmulh v24.4s, v24.4s, v14.4s\n"
        "sqdmulh v25.4s, v25.4s, v15.4s\n"
        "sqdmulh v26.4s, v26.4s, v14.4s\n"
        "sqdmulh v27.4s, v27.4s, v15.4s\n"
        "sqdmulh v28.4s, v28.4s, v14.4s\n"
        "sqdmulh v29.4s, v29.4s, v15.4s\n"
        "sqdmulh v30.4s, v30.4s, v14.4s\n"
        "sqdmulh v31.4s, v31.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"
        "srshl v20.4s, v20.4s, v11.4s\n"
        "srshl v21.4s, v21.4s, v12.4s\n"
        "srshl v22.4s, v22.4s, v11.4s\n"
        "srshl v23.4s, v23.4s, v12.4s\n"
        "srshl v24.4s, v24.4s, v11.4s\n"
        "srshl v25.4s, v25.4s, v12.4s\n"
        "srshl v26.4s, v26.4s, v11.4s\n"
        "srshl v27.4s, v27.4s, v12.4s\n"
        "srshl v28.4s, v28.4s, v11.4s\n"
        "srshl v29.4s, v29.4s, v12.4s\n"
        "srshl v30.4s, v30.4s, v11.4s\n"
        "srshl v31.4s, v31.4s, v12.4s\n"
        "b 9f\n"

        "8:\n"
        // Case where channels are columns

        // Apply the positive exponent part of the multiplier.
        "dup v4.4s, v9.s[0]\n"
        "dup v5.4s, v9.s[1]\n"
        "dup v6.4s, v9.s[2]\n"
        "dup v7.4s, v9.s[3]\n"
        "sshl v16.4s, v16.4s, v4.4s\n"
        "sshl v17.4s, v17.4s, v4.4s\n"
        "sshl v18.4s, v18.4s, v5.4s\n"
        "sshl v19.4s, v19.4s, v5.4s\n"
        "sshl v20.4s, v20.4s, v6.4s\n"
        "sshl v21.4s, v21.4s, v6.4s\n"
        "sshl v22.4s, v22.4s, v7.4s\n"
        "sshl v23.4s, v23.4s, v7.4s\n"
        "dup v4.4s, v10.s[0]\n"
        "dup v5.4s, v10.s[1]\n"
        "dup v6.4s, v10.s[2]\n"
        "dup v7.4s, v10.s[3]\n"
        "sshl v24.4s, v24.4s, v4.4s\n"
        "sshl v25.4s, v25.4s, v4.4s\n"
        "sshl v26.4s, v26.4s, v5.4s\n"
        "sshl v27.4s, v27.4s, v5.4s\n"
        "sshl v28.4s, v28.4s, v6.4s\n"
        "sshl v29.4s, v29.4s, v6.4s\n"
        "sshl v30.4s, v30.4s, v7.4s\n"
        "sshl v31.4s, v31.4s, v7.4s\n"
        "11:\n"

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v14.s[0]\n"
        "sqdmulh v17.4s, v17.4s, v14.s[0]\n"
        "sqdmulh v18.4s, v18.4s, v14.s[1]\n"
        "sqdmulh v19.4s, v19.4s, v14.s[1]\n"
        "sqdmulh v20.4s, v20.4s, v14.s[2]\n"
        "sqdmulh v21.4s, v21.4s, v14.s[2]\n"
        "sqdmulh v22.4s, v22.4s, v14.s[3]\n"
        "sqdmulh v23.4s, v23.4s, v14.s[3]\n"
        "sqdmulh v24.4s, v24.4s, v15.s[0]\n"
        "sqdmulh v25.4s, v25.4s, v15.s[0]\n"
        "sqdmulh v26.4s, v26.4s, v15.s[1]\n"
        "sqdmulh v27.4s, v27.4s, v15.s[1]\n"
        "sqdmulh v28.4s, v28.4s, v15.s[2]\n"
        "sqdmulh v29.4s, v29.4s, v15.s[2]\n"
        "sqdmulh v30.4s, v30.4s, v15.s[3]\n"
        "sqdmulh v31.4s, v31.4s, v15.s[3]\n"

        // Apply the negative exponent part of the multiplier.
        "dup v4.4s, v11.s[0]\n"
        "dup v5.4s, v11.s[1]\n"
        "dup v6.4s, v11.s[2]\n"
        "dup v7.4s, v11.s[3]\n"
        "srshl v16.4s, v16.4s, v4.4s\n"
        "srshl v17.4s, v17.4s, v4.4s\n"
        "srshl v18.4s, v18.4s, v5.4s\n"
        "srshl v19.4s, v19.4s, v5.4s\n"
        "srshl v20.4s, v20.4s, v6.4s\n"
        "srshl v21.4s, v21.4s, v6.4s\n"
        "srshl v22.4s, v22.4s, v7.4s\n"
        "srshl v23.4s, v23.4s, v7.4s\n"
        "dup v4.4s, v12.s[0]\n"
        "dup v5.4s, v12.s[1]\n"
        "dup v6.4s, v12.s[2]\n"
        "dup v7.4s, v12.s[3]\n"
        "srshl v24.4s, v24.4s, v4.4s\n"
        "srshl v25.4s, v25.4s, v4.4s\n"
        "srshl v26.4s, v26.4s, v5.4s\n"
        "srshl v27.4s, v27.4s, v5.4s\n"
        "srshl v28.4s, v28.4s, v6.4s\n"
        "srshl v29.4s, v29.4s, v6.4s\n"
        "srshl v30.4s, v30.4s, v7.4s\n"
        "srshl v31.4s, v31.4s, v7.4s\n"
        "9:\n"

        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"
        "sqadd v18.8h, v18.8h, v14.8h\n"
        "sqadd v19.8h, v19.8h, v14.8h\n"
        "sqadd v20.8h, v20.8h, v14.8h\n"
        "sqadd v21.8h, v21.8h, v14.8h\n"
        "sqadd v22.8h, v22.8h, v14.8h\n"
        "sqadd v23.8h, v23.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "sqxtun2 v16.16b, v17.8h\n"
        "sqxtun v17.8b, v18.8h\n"
        "sqxtun2 v17.16b, v19.8h\n"
        "sqxtun v18.8b, v20.8h\n"
        "sqxtun2 v18.16b, v21.8h\n"
        "sqxtun v19.8b, v22.8h\n"
        "sqxtun2 v19.16b, v23.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        "umax v17.16b, v17.16b, v14.16b\n"
        "umax v18.16b, v18.16b, v14.16b\n"
        "umax v19.16b, v19.16b, v14.16b\n"

        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        "umin v17.16b, v17.16b, v15.16b\n"
        "umin v18.16b, v18.16b, v15.16b\n"
        "umin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"
        "sqadd v18.8h, v18.8h, v14.8h\n"
        "sqadd v19.8h, v19.8h, v14.8h\n"
        "sqadd v20.8h, v20.8h, v14.8h\n"
        "sqadd v21.8h, v21.8h, v14.8h\n"
        "sqadd v22.8h, v22.8h, v14.8h\n"
        "sqadd v23.8h, v23.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        "sqxtn2 v16.16b, v17.8h\n"
        "sqxtn v17.8b, v18.8h\n"
        "sqxtn2 v17.16b, v19.8h\n"
        "sqxtn v18.8b, v20.8h\n"
        "sqxtn2 v18.16b, v21.8h\n"
        "sqxtn v19.8b, v22.8h\n"
        "sqxtn2 v19.16b, v23.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        "smax v17.16b, v17.16b, v14.16b\n"
        "smax v18.16b, v18.16b, v14.16b\n"
        "smax v19.16b, v19.16b, v14.16b\n"

        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        "smin v17.16b, v17.16b, v15.16b\n"
        "smin v18.16b, v18.16b, v15.16b\n"
        "smin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 150b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"
        "saddw v20.4s, v20.4s, v14.4h\n"
        "saddw v21.4s, v21.4s, v14.4h\n"
        "saddw v22.4s, v22.4s, v14.4h\n"
        "saddw v23.4s, v23.4s, v14.4h\n"
        "saddw v24.4s, v24.4s, v14.4h\n"
        "saddw v25.4s, v25.4s, v14.4h\n"
        "saddw v26.4s, v26.4s, v14.4h\n"
        "saddw v27.4s, v27.4s, v14.4h\n"
        "saddw v28.4s, v28.4s, v14.4h\n"
        "saddw v29.4s, v29.4s, v14.4h\n"
        "saddw v30.4s, v30.4s, v14.4h\n"
        "saddw v31.4s, v31.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        "smax v18.8h, v18.8h, v14.8h\n"
        "smax v19.8h, v19.8h, v14.8h\n"
        "smax v20.8h, v20.8h, v14.8h\n"
        "smax v21.8h, v21.8h, v14.8h\n"
        "smax v22.8h, v22.8h, v14.8h\n"
        "smax v23.8h, v23.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        "smin v18.8h, v18.8h, v15.8h\n"
        "smin v19.8h, v19.8h, v15.8h\n"
        "smin v20.8h, v20.8h, v15.8h\n"
        "smin v21.8h, v21.8h, v15.8h\n"
        "smin v22.8h, v22.8h, v15.8h\n"
        "smin v23.8h, v23.8h, v15.8h\n"

        // Compute how much of the 8x8 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 16bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 250b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x8 block of destination 32it values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x8 block fits.
        // Write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "st1 {v16.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v23)
        "st1 {v24.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v24)
        "st1 {v25.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v25)
        "st1 {v26.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v26)
        "st1 {v27.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v27)
        "st1 {v28.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v28)
        "st1 {v29.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v29)
        "st1 {v30.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v30)
        "st1 {v31.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v31)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x8 block fits.
        "mov x4, %[dst_ptr]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v16.4s, v17.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v18.4s, v19.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v20.4s, v21.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v22.4s, v23.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v24.4s, v25.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v26.4s, v27.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v28.4s, v29.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "add x4, x4, x11\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov x3, x4\n"
        "st1 {v30.4s, v31.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 350b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}


// Similar to the above 8-bit dotprod kernel, but specialized for the case of
// RHS cols == 1.
// Relevant target CPUs for this kernel include ARM Cortex-A76,
// since these are 64-bit, out-of-order and with dotprod support.
void Kernel8bitNeonDotprod1Col(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprod)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  RUY_DCHECK(!(params.flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL));

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v15 are used to load int8 data from LHS and
  // RHS. At least v0 and v1 are used to load a 8x4 block of LHS, and v2 and
  // v3 are used to load a 4x8 block of RHS, like this:
  //
  //                            int8 RHS 4x1 block
  //                           /-------|
  //                           |v2.b[0]|
  //                           |  ...  |
  //                           |v2.b[3]|
  //                           \-------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /--------|
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]|
  //  |  ...          ...   |  |  ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]|
  //  |  ...         ...    |  |  ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]|
  //  \---------------------/  \--------/
  //                           int32 accumulators 8x1 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for loading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.8b}, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Ordinary kernel inner loop (over depth), the simpler loop that the
        // above was an equivalent 4x-partially-unrolled version of.

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Because of the data that we have already loaded, we can start the
        // loop body right away with some multiply-adds.
        // Each iteration of this loop advances by 4 levels of depth.
        "add w1, w1, #4\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        // Loop termination condition.
        "cmp w1, w12\n"
        "ld1 {v2.8b}, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

        "blt 2b\n"

        "79:\n"
        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last 4 levels of depth, for which the LHS
        // and RHS data is already loaded.

        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        "add x5, x1, %x[row], lsl #2\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.8b}, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3], #16\n"
        "ld1 {v15.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2], #16\n"
        "ld1 {v12.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"

        "smin v11.4s, v8.4s, v9.4s\n"
        "smin v12.4s, v8.4s, v10.4s\n"
        "sub v9.4s, v9.4s, v11.4s\n"
        "sub v10.4s, v10.4s, v12.4s\n"

        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v9.4s\n"
        "sshl v17.4s, v17.4s, v10.4s\n"
        "403:\n"

        "ldr q14, [x4]\n" // multiplier_fixedpoint
        "ldr q15, [x4, #16]\n" // multiplier_fixedpoint

        // Apply the fixed-point part of the multiplier.
        "sqdmulh v16.4s, v16.4s, v14.4s\n"
        "sqdmulh v17.4s, v17.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        // All data in v16 at this point.

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8, leaving all data in the
        // lower half of v16.
        "sqxtun v16.8b, v16.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"

        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"

        // Compute how much of the 8x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x1, there are some 8x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"

        // Test if w1==8, i.e. if all of the 8x1 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v16.8b}, [x3]\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"


        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "sqadd v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"

        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"

        // Compute how much of the 8x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"

        // Test if w1==8, i.e. if all of the 8x1 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v16.8b}, [x3]\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"

        // Compute how much of the 8x1 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"

        // Test if w1==8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 16bit values to the destination
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v16.8h}, [x3]\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x1 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x1 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x1 block of destination 32 bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x1, there are some 8x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x4, %[dst_ptr]\n"
        "mov x3, x4\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.4s, v17.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17");
}

// Variant of the above Kernel8bitNeonDotprod, tuned for in-order
// CPUs. Specifically here, the relevant in-order CPUs are ARM Cortex-A55r1,
// since these are 64-bit and support dotprod.
//
// While this kernel does not have a direct equivalent in gemmlowp, it was
// developed based on insights that David Mansell at ARM shared with their
// contribution of gemmlowp kernels tuned for Cortex-A55r1, with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A55r1:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4412
void Kernel8bitNeonDotprodA55ish(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label(
      "Kernel (kNeonDotprod, optimized for in-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr =
      static_cast<const int8_t*>(params.rhs_base_ptr);
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // RHS.
  //
  //                                      int8 RHS 4x8 block
  //                           /-----------------------------------------|
  //                           |v2.b[0] ... v2.b[12] v3.b[0] ... v3.b[12]|
  //                           |  ...                              ...   |
  //                           |v2.b[3] ... v2.b[15] v3.b[3] ... v3.b[15]|
  //                           \-----------------------------------------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /-----------------------------------------|
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]           ...           v30.s[0]|
  //  |  ...          ...   |  |  ...                              ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]           ...           v30.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]           ...           v31.s[0]|
  //  |  ...         ...    |  |  ...                              ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 8x8 block
  //
  // There is no RUY_OPT_MAX_STREAMING 4x-unrolled part in this kernel because
  // we did not observe a benefit of such partial unrolling on in-order CPUs.
  //
  // v4 -- v7 are unused, and v8 -- v15 are used for loading parameters used for
  // the post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        RUY_MAKE_ZERO(v16)
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        RUY_MAKE_ZERO(v17)
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        RUY_MAKE_ZERO(v18)
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        RUY_MAKE_ZERO(v19)
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v20)
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v21)
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        RUY_MAKE_ZERO(v22)
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        RUY_MAKE_ZERO(v28)
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        RUY_MAKE_ZERO(v29)
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        RUY_MAKE_ZERO(v30)
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        RUY_MAKE_ZERO(v31)


        "1:\n"

        "add x5, %[lhs_ptr], x12, lsl #3\n"
        "sub x5, x5, #32\n"
        "cmp %[lhs_ptr], x5\n"

        "beq 79f\n"

        // Main accumulation loop
        "2:\n"
        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        "ldr x1, [%[lhs_ptr], #8]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        "ldr x3, [%[rhs_ptr], #8]\n"
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        "ldr x4, [%[rhs_ptr], #24]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ldr d0, [%[lhs_ptr], #0]\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        "ins v0.d[1], x1\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        "add %[lhs_ptr], %[lhs_ptr], #32\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ldr d2, [%[rhs_ptr], #0]\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        "ins v2.d[1], x3\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        "cmp %[lhs_ptr], x5\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ldr d3, [%[rhs_ptr], #-16]\n"
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        "ldr d1, [%[lhs_ptr], #-16]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        "ins v3.d[1], x4\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        "ins v1.d[1], x2\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "blt 2b\n"

        // Last accumulation steps, nothing left to load.
        "79:\n"
        ".inst 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        ".inst 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        "cmp %w[row], w7\n"  // Have we finished the last row?
        ".inst 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".inst 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        ".inst 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".inst 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".inst 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".inst 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        ".inst 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".inst 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".inst 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".inst 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        // Load some parameters needed for the end work on current block.
        "mvni v8.4s, #0\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        // Determine the channel index.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.2s}, [x1], #8\n"
        "ldr x5, [x1], #8\n"
        "ins v14.d[1], x5\n"
        "ld1 {v15.2s}, [x1], #8\n"
        "ldr x5, [x1], #8\n"
        "ins v15.d[1], x5\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"
        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v15.4s\n"
        "add v20.4s, v20.4s, v14.4s\n"
        "add v21.4s, v21.4s, v15.4s\n"
        "add v22.4s, v22.4s, v14.4s\n"
        "add v23.4s, v23.4s, v15.4s\n"
        "add v24.4s, v24.4s, v14.4s\n"
        "add v25.4s, v25.4s, v15.4s\n"
        "add v26.4s, v26.4s, v14.4s\n"
        "add v27.4s, v27.4s, v15.4s\n"
        "add v28.4s, v28.4s, v14.4s\n"
        "add v29.4s, v29.4s, v15.4s\n"
        "add v30.4s, v30.4s, v14.4s\n"
        "add v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v10.4s, v14.s[0]\n"
        "dup v11.4s, v14.s[1]\n"
        "add v16.4s, v16.4s, v10.4s\n"
        "dup v12.4s, v14.s[2]\n"
        "add v17.4s, v17.4s, v10.4s\n"
        "dup v13.4s, v14.s[3]\n"
        "add v18.4s, v18.4s, v11.4s\n"
        "dup v10.4s, v15.s[0]\n"
        "add v19.4s, v19.4s, v11.4s\n"
        "dup v11.4s, v15.s[1]\n"
        "add v20.4s, v20.4s, v12.4s\n"
        "add v21.4s, v21.4s, v12.4s\n"
        "dup v12.4s, v15.s[2]\n"
        "add v22.4s, v22.4s, v13.4s\n"
        "add v23.4s, v23.4s, v13.4s\n"
        "dup v13.4s, v15.s[3]\n"
        "add v24.4s, v24.4s, v10.4s\n"
        "add v25.4s, v25.4s, v10.4s\n"
        "add v26.4s, v26.4s, v11.4s\n"
        "add v27.4s, v27.4s, v11.4s\n"
        "add v28.4s, v28.4s, v12.4s\n"
        "add v29.4s, v29.4s, v12.4s\n"
        "add v30.4s, v30.4s, v13.4s\n"
        "add v31.4s, v31.4s, v13.4s\n"
        "7:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x5, x5, %x[col], lsl #2\n"
        // Load 8 rhs_sums values.
        "ld1 {v14.2s}, [x5], #8\n"
        "ldr x7, [x5], #8\n"
        "ld1 {v15.2s}, [x5], #8\n"
        "ins v14.d[1], x7\n"
        "ldr x7, [x5], #8\n"
        "ins v15.d[1], x7\n"
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "mls v18.4s, v10.4s, v14.s[1]\n"
        "mls v19.4s, v10.4s, v14.s[1]\n"
        "mls v20.4s, v10.4s, v14.s[2]\n"
        "mls v21.4s, v10.4s, v14.s[2]\n"
        "mls v22.4s, v10.4s, v14.s[3]\n"
        "mls v23.4s, v10.4s, v14.s[3]\n"
        "mls v24.4s, v10.4s, v15.s[0]\n"
        "mls v25.4s, v10.4s, v15.s[0]\n"
        "mls v26.4s, v10.4s, v15.s[1]\n"
        "mls v27.4s, v10.4s, v15.s[1]\n"
        "mls v28.4s, v10.4s, v15.s[2]\n"
        "mls v29.4s, v10.4s, v15.s[2]\n"
        "mls v30.4s, v10.4s, v15.s[3]\n"
        "mls v31.4s, v10.4s, v15.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Load 8 lhs_sums values.
        "ld1 {v11.2s}, [x2], #8\n"
        "ldr x4, [x2], #8\n"
        "ins v11.d[1], x4\n"
        "ld1 {v12.2s}, [x2], #8\n"
        "ldr x4, [x2], #8\n"
        "ins v12.d[1], x4\n"
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v12.4s\n"
        "sub v20.4s, v20.4s, v11.4s\n"
        "sub v21.4s, v21.4s, v12.4s\n"
        "sub v22.4s, v22.4s, v11.4s\n"
        "sub v23.4s, v23.4s, v12.4s\n"
        "sub v24.4s, v24.4s, v11.4s\n"
        "sub v25.4s, v25.4s, v12.4s\n"
        "sub v26.4s, v26.4s, v11.4s\n"
        "sub v27.4s, v27.4s, v12.4s\n"
        "sub v28.4s, v28.4s, v11.4s\n"
        "sub v29.4s, v29.4s, v12.4s\n"
        "sub v30.4s, v30.4s, v11.4s\n"
        "sub v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        // Compute the multiplier_exponent pointer
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, x3, lsl #2\n"
        "csel x1, x1, x5, eq\n"
        // Load multiplier_exponent
        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"
        // Separate positive and negative exponents
        "smin v11.4s, v8.4s, v9.4s\n"
        "smin v12.4s, v8.4s, v10.4s\n"
        "sub v9.4s, v9.4s, v11.4s\n"
        "sub v10.4s, v10.4s, v12.4s\n"

        // Compute the multiplier_fixedpoint pointer
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "add x5, x4, x3, lsl #2\n"
        "csel x4, x4, x5, eq\n"
        // Load multiplier_fixedpoint
        "ldr q14, [x4]\n"
        "ldr q15, [x4, #16]\n"

        // Jump based on channel dimension.
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 8f\n"
        // Case where channels are rows

        // Apply the positive exponent part of the multiplier.
        "sshl v16.4s, v16.4s, v9.4s\n"
        "sshl v17.4s, v17.4s, v10.4s\n"
        "sshl v18.4s, v18.4s, v9.4s\n"
        "sshl v19.4s, v19.4s, v10.4s\n"
        "sshl v20.4s, v20.4s, v9.4s\n"
        "sshl v21.4s, v21.4s, v10.4s\n"
        "sshl v22.4s, v22.4s, v9.4s\n"
        "sshl v23.4s, v23.4s, v10.4s\n"
        "sshl v24.4s, v24.4s, v9.4s\n"
        "sshl v25.4s, v25.4s, v10.4s\n"
        "sshl v26.4s, v26.4s, v9.4s\n"
        "sshl v27.4s, v27.4s, v10.4s\n"
        "sshl v28.4s, v28.4s, v9.4s\n"
        "sshl v29.4s, v29.4s, v10.4s\n"
        "sshl v30.4s, v30.4s, v9.4s\n"
        "sshl v31.4s, v31.4s, v10.4s\n"
        "10:\n"

        // Apply the fixed-point part of the multiplier.
        //
        // ... and, interleaved into that:
        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "sqdmulh v16.4s, v16.4s, v14.4s\n"
        "ldr x1, [%[lhs_ptr]], #8\n"
        "sqdmulh v17.4s, v17.4s, v15.4s\n"
        "ld1 {v1.8b}, [%[lhs_ptr]], #8\n"
        "sqdmulh v18.4s, v18.4s, v14.4s\n"
        "ldr x2, [%[lhs_ptr]], #8\n"
        "sqdmulh v19.4s, v19.4s, v15.4s\n"
        "ld1 {v2.8b}, [%[rhs_ptr]], #8\n"
        "sqdmulh v20.4s, v20.4s, v14.4s\n"
        "ldr x5, [%[rhs_ptr]], #8\n"
        "sqdmulh v21.4s, v21.4s, v15.4s\n"
        "ld1 {v3.8b}, [%[rhs_ptr]], #8\n"
        "sqdmulh v22.4s, v22.4s, v14.4s\n"
        "ldr x6, [%[rhs_ptr]], #8\n"
        "sqdmulh v23.4s, v23.4s, v15.4s\n"
        "sqdmulh v24.4s, v24.4s, v14.4s\n"
        "sqdmulh v25.4s, v25.4s, v15.4s\n"
        "sqdmulh v26.4s, v26.4s, v14.4s\n"
        "sqdmulh v27.4s, v27.4s, v15.4s\n"
        "sqdmulh v28.4s, v28.4s, v14.4s\n"
        "sqdmulh v29.4s, v29.4s, v15.4s\n"
        "sqdmulh v30.4s, v30.4s, v14.4s\n"
        "sqdmulh v31.4s, v31.4s, v15.4s\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"
        "srshl v20.4s, v20.4s, v11.4s\n"
        "srshl v21.4s, v21.4s, v12.4s\n"
        "srshl v22.4s, v22.4s, v11.4s\n"
        "srshl v23.4s, v23.4s, v12.4s\n"
        "srshl v24.4s, v24.4s, v11.4s\n"
        "srshl v25.4s, v25.4s, v12.4s\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "srshl v26.4s, v26.4s, v11.4s\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "srshl v27.4s, v27.4s, v12.4s\n"
        "ins v0.d[1], x1\n"
        "srshl v28.4s, v28.4s, v11.4s\n"
        "ins v1.d[1], x2\n"
        "srshl v29.4s, v29.4s, v12.4s\n"
        "ins v2.d[1], x5\n"
        "srshl v30.4s, v30.4s, v11.4s\n"
        "ins v3.d[1], x6\n"
        "srshl v31.4s, v31.4s, v12.4s\n"
        "b 9f\n"

        "8:\n"
        // Case where channels are columns

        // Apply the positive exponent part of the multiplier.
        "dup v4.4s, v9.s[0]\n"
        "dup v5.4s, v9.s[1]\n"
        "sshl v16.4s, v16.4s, v4.4s\n"
        "dup v6.4s, v9.s[2]\n"
        "sshl v17.4s, v17.4s, v4.4s\n"
        "dup v7.4s, v9.s[3]\n"
        "sshl v18.4s, v18.4s, v5.4s\n"
        "dup v4.4s, v10.s[0]\n"
        "sshl v19.4s, v19.4s, v5.4s\n"
        "dup v5.4s, v10.s[1]\n"
        "sshl v20.4s, v20.4s, v6.4s\n"
        "sshl v21.4s, v21.4s, v6.4s\n"
        "dup v6.4s, v10.s[2]\n"
        "sshl v22.4s, v22.4s, v7.4s\n"
        "sshl v23.4s, v23.4s, v7.4s\n"
        "dup v7.4s, v10.s[3]\n"
        "sshl v24.4s, v24.4s, v4.4s\n"
        "sshl v25.4s, v25.4s, v4.4s\n"
        "sshl v26.4s, v26.4s, v5.4s\n"
        "sshl v27.4s, v27.4s, v5.4s\n"
        "sshl v28.4s, v28.4s, v6.4s\n"
        "sshl v29.4s, v29.4s, v6.4s\n"
        "sshl v30.4s, v30.4s, v7.4s\n"
        "sshl v31.4s, v31.4s, v7.4s\n"
        "11:\n"

        // Apply the fixed-point part of the multiplier.
        //
        // ... and, interleaved into that:
        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "sqdmulh v16.4s, v16.4s, v14.s[0]\n"
        "ldr x1, [%[lhs_ptr]], #8\n"
        "sqdmulh v17.4s, v17.4s, v14.s[0]\n"
        "ld1 {v1.8b}, [%[lhs_ptr]], #8\n"
        "sqdmulh v18.4s, v18.4s, v14.s[1]\n"
        "ldr x2, [%[lhs_ptr]], #8\n"
        "sqdmulh v19.4s, v19.4s, v14.s[1]\n"
        "ld1 {v2.8b}, [%[rhs_ptr]], #8\n"
        "sqdmulh v20.4s, v20.4s, v14.s[2]\n"
        "ldr x5, [%[rhs_ptr]], #8\n"
        "sqdmulh v21.4s, v21.4s, v14.s[2]\n"
        "ld1 {v3.8b}, [%[rhs_ptr]], #8\n"
        "sqdmulh v22.4s, v22.4s, v14.s[3]\n"
        "ldr x6, [%[rhs_ptr]], #8\n"
        "sqdmulh v23.4s, v23.4s, v14.s[3]\n"
        "dup v4.4s, v11.s[0]\n"
        "sqdmulh v24.4s, v24.4s, v15.s[0]\n"
        "dup v5.4s, v11.s[1]\n"
        "sqdmulh v25.4s, v25.4s, v15.s[0]\n"
        "dup v6.4s, v11.s[2]\n"
        "sqdmulh v26.4s, v26.4s, v15.s[1]\n"
        "dup v7.4s, v11.s[3]\n"
        "sqdmulh v27.4s, v27.4s, v15.s[1]\n"
        "sqdmulh v28.4s, v28.4s, v15.s[2]\n"
        "sqdmulh v29.4s, v29.4s, v15.s[2]\n"
        "sqdmulh v30.4s, v30.4s, v15.s[3]\n"
        "sqdmulh v31.4s, v31.4s, v15.s[3]\n"

        // Apply the negative exponent part of the multiplier.
        "srshl v16.4s, v16.4s, v4.4s\n"
        "srshl v17.4s, v17.4s, v4.4s\n"
        "dup v4.4s, v12.s[0]\n"
        "srshl v18.4s, v18.4s, v5.4s\n"
        "srshl v19.4s, v19.4s, v5.4s\n"
        "dup v5.4s, v12.s[1]\n"
        "srshl v20.4s, v20.4s, v6.4s\n"
        "srshl v21.4s, v21.4s, v6.4s\n"
        "dup v6.4s, v12.s[2]\n"
        "srshl v22.4s, v22.4s, v7.4s\n"
        "srshl v23.4s, v23.4s, v7.4s\n"
        "dup v7.4s, v12.s[3]\n"
        "srshl v24.4s, v24.4s, v4.4s\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "srshl v25.4s, v25.4s, v4.4s\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "srshl v26.4s, v26.4s, v5.4s\n"
        "ins v0.d[1], x1\n"
        "srshl v27.4s, v27.4s, v5.4s\n"
        "ins v1.d[1], x2\n"
        "srshl v28.4s, v28.4s, v6.4s\n"
        "ins v2.d[1], x5\n"
        "srshl v29.4s, v29.4s, v6.4s\n"
        "ins v3.d[1], x6\n"
        "srshl v30.4s, v30.4s, v7.4s\n"
        "srshl v31.4s, v31.4s, v7.4s\n"
        "9:\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // Destination zero_point
        "dup v14.8h, v13.h[4]\n"
        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"
        "sqadd v18.8h, v18.8h, v14.8h\n"
        "sqadd v19.8h, v19.8h, v14.8h\n"
        "sqadd v20.8h, v20.8h, v14.8h\n"
        "sqadd v21.8h, v21.8h, v14.8h\n"
        "sqadd v22.8h, v22.8h, v14.8h\n"
        "sqadd v23.8h, v23.8h, v14.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "sqxtun2 v16.16b, v17.8h\n"
        "sqxtun v17.8b, v18.8h\n"
        "sqxtun2 v17.16b, v19.8h\n"
        "sqxtun v18.8b, v20.8h\n"
        "sqxtun2 v18.16b, v21.8h\n"
        "sqxtun v19.8b, v22.8h\n"
        "sqxtun2 v19.16b, v23.8h\n"

        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "umax v17.16b, v17.16b, v14.16b\n"
        "mov w3, #8\n"
        "umax v18.16b, v18.16b, v14.16b\n"
        "cmp w1, #8\n"
        "umax v19.16b, v19.16b, v14.16b\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        "cmp w2, #8\n"
        "umin v17.16b, v17.16b, v15.16b\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"
        "umin v18.16b, v18.16b, v15.16b\n"
        "umin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"

        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // Destination zero_point
        "dup v14.8h, v13.h[4]\n"
        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "sqadd v16.8h, v16.8h, v14.8h\n"
        "sqadd v17.8h, v17.8h, v14.8h\n"
        "sqadd v18.8h, v18.8h, v14.8h\n"
        "sqadd v19.8h, v19.8h, v14.8h\n"
        "sqadd v20.8h, v20.8h, v14.8h\n"
        "sqadd v21.8h, v21.8h, v14.8h\n"
        "sqadd v22.8h, v22.8h, v14.8h\n"
        "sqadd v23.8h, v23.8h, v14.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "sqxtn2 v16.16b, v17.8h\n"
        "sqxtn v17.8b, v18.8h\n"
        "sqxtn2 v17.16b, v19.8h\n"
        "sqxtn v18.8b, v20.8h\n"
        "sqxtn2 v18.16b, v21.8h\n"
        "sqxtn v19.8b, v22.8h\n"
        "sqxtn2 v19.16b, v23.8h\n"

        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "smax v17.16b, v17.16b, v14.16b\n"
        "mov w3, #8\n"
        "smax v18.16b, v18.16b, v14.16b\n"
        "cmp w1, #8\n"
        "smax v19.16b, v19.16b, v14.16b\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        "cmp w2, #8\n"
        "smin v17.16b, v17.16b, v15.16b\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"
        "smin v18.16b, v18.16b, v15.16b\n"
        "smin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 150b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"

        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"
        "saddw v20.4s, v20.4s, v14.4h\n"
        "saddw v21.4s, v21.4s, v14.4h\n"
        "saddw v22.4s, v22.4s, v14.4h\n"
        "saddw v23.4s, v23.4s, v14.4h\n"
        "saddw v24.4s, v24.4s, v14.4h\n"
        "saddw v25.4s, v25.4s, v14.4h\n"
        "saddw v26.4s, v26.4s, v14.4h\n"
        "saddw v27.4s, v27.4s, v14.4h\n"
        "saddw v28.4s, v28.4s, v14.4h\n"
        "saddw v29.4s, v29.4s, v14.4h\n"
        "saddw v30.4s, v30.4s, v14.4h\n"
        "saddw v31.4s, v31.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        "smax v18.8h, v18.8h, v14.8h\n"
        "smax v19.8h, v19.8h, v14.8h\n"
        "smax v20.8h, v20.8h, v14.8h\n"
        "smax v21.8h, v21.8h, v14.8h\n"
        "smax v22.8h, v22.8h, v14.8h\n"
        "smax v23.8h, v23.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        "smin v18.8h, v18.8h, v15.8h\n"
        "smin v19.8h, v19.8h, v15.8h\n"
        "smin v20.8h, v20.8h, v15.8h\n"
        "smin v21.8h, v21.8h, v15.8h\n"
        "smin v22.8h, v22.8h, v15.8h\n"
        "smin v23.8h, v23.8h, v15.8h\n"

        // Compute how much of the 8x8 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v16.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v17.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v18.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v19.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v20.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v21.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v22.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "st1 {v23.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 250b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "ldr x1, [%[lhs_ptr]], #8\n"
        "ld1 {v1.8b}, [%[lhs_ptr]], #8\n"
        "ldr x2, [%[lhs_ptr]], #8\n"
        "ld1 {v2.8b}, [%[rhs_ptr]], #8\n"
        "ldr x5, [%[rhs_ptr]], #8\n"
        "ld1 {v3.8b}, [%[rhs_ptr]], #8\n"
        "ldr x6, [%[rhs_ptr]], #8\n"
        "ins v0.d[1], x1\n"
        "ins v1.d[1], x2\n"
        "ins v2.d[1], x5\n"
        "ins v3.d[1], x6\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x8 block of destination 32it values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x8 block fits.
        // Write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "st1 {v16.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v23)
        "st1 {v24.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v24)
        "st1 {v25.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v25)
        "st1 {v26.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v26)
        "st1 {v27.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v27)
        "st1 {v28.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v28)
        "st1 {v29.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v29)
        "st1 {v30.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v30)
        "st1 {v31.4s}, [x3], #16\n"
        RUY_MAKE_ZERO(v31)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x8 block fits.
        "mov x4, %[dst_ptr]\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v16.4s, v17.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v18.4s, v19.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v20.4s, v21.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v22.4s, v23.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v24.4s, v25.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v26.4s, v27.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v28.4s, v29.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "st1 {v30.4s, v31.4s}, [x4], x11\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".inst 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 350b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}
#undef RUY_OFFSET_BIAS
#undef RUY_OFFSET_LHS_SUMS
#undef RUY_OFFSET_RHS_SUMS
#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_MULTIPLIER_FIXEDPOINT
#undef RUY_OFFSET_MULTIPLIER_EXPONENT
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR
#undef RUY_OFFSET_LHS_ZERO_POINT
#undef RUY_OFFSET_RHS_ZERO_POINT
#undef RUY_OFFSET_DST_ZERO_POINT
#undef RUY_OFFSET_PROD_ZP_DEPTH
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_DST_ROWS
#undef RUY_OFFSET_DST_COLS
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_FLAGS

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 8
#define RUY_OFFSET_DST_BASE_PTR 16
#define RUY_OFFSET_BIAS 24
#define RUY_OFFSET_START_ROW 32
#define RUY_OFFSET_START_COL 36
#define RUY_OFFSET_LAST_ROW 40
#define RUY_OFFSET_LAST_COL 44
#define RUY_OFFSET_LHS_STRIDE 56
#define RUY_OFFSET_RHS_STRIDE 60
#define RUY_OFFSET_DST_STRIDE 64
#define RUY_OFFSET_DEPTH 68
#define RUY_OFFSET_CLAMP_MIN 72
#define RUY_OFFSET_CLAMP_MAX 76
#define RUY_OFFSET_FLAGS 80

template <typename Params>
void CheckOffsetsInKernelParamsFloat(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, bias) == RUY_OFFSET_BIAS, "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, start_col) == RUY_OFFSET_START_COL, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
}

// Just a plain float kernel; good enough for out-of-order cores.
// The closest to it in the gemmlowp collection would be
// NEON_64bit_GEMM_Float32_WithScalar,
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L3925
//
// Besides ruy-ification, the main nuance here is that we stick to a 8x8
// width instead of the wider 12x8 that the register space permits and that
// the aforementioned gemmlowp kernel uses.  Ruy likes powers of two for now
// and we don't have evidence that going beyond 8x8 is needed.
void KernelFloatNeon(const KernelParamsFloat<8, 8>& params) {
  CheckOffsetsInKernelParamsFloat(params);
  profiler::ScopeLabel label("Kernel (kNeon)");

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v15 are used to load data from LHS and RHS.
  // At least v0 and v1 are used to load a 8x1 block of LHS, and v2 and
  // v3 are used to load a 1x8 block of RHS, like this:
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------|
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------|
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for floading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

#if RUY_OPT(MAX_STREAMING)
        "cmp w12, #8\n"
        "blt 78f\n"
        "and w2, w12, #-4\n"

        "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v5.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v6.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.4s}, [%[rhs_ptr]], #16\n"

        "ld1 {v8.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v9.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v10.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v11.4s}, [%[rhs_ptr]], #16\n"

        "ld1 {v12.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v13.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v14.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v15.4s}, [%[rhs_ptr]], #16\n"
        "mov w1, #4\n"

        "80:\n"

        "add %[lhs_ptr], %[lhs_ptr], #128\n"
        "add %[rhs_ptr], %[rhs_ptr], #128\n"

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ldr q0, [%[lhs_ptr], #-128]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr q3, [%[rhs_ptr], #-112]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "ldr q1, [%[lhs_ptr], #-112]\n"
        "fmla v16.4s, v4.4s, v6.s[0]\n"
        "fmla v18.4s, v4.4s, v6.s[1]\n"
        "ldr q2, [%[rhs_ptr], #-128]\n"
        "fmla v20.4s, v4.4s, v6.s[2]\n"
        "fmla v22.4s, v4.4s, v6.s[3]\n"

        "fmla v24.4s, v4.4s, v7.s[0]\n"
        "fmla v26.4s, v4.4s, v7.s[1]\n"
        "fmla v28.4s, v4.4s, v7.s[2]\n"
        "fmla v30.4s, v4.4s, v7.s[3]\n"
        "ldr q4, [%[lhs_ptr], #-96]\n"
        "fmla v25.4s, v5.4s, v7.s[0]\n"
        "fmla v27.4s, v5.4s, v7.s[1]\n"
        "fmla v29.4s, v5.4s, v7.s[2]\n"
        "fmla v31.4s, v5.4s, v7.s[3]\n"
        "ldr q7, [%[rhs_ptr], #-80]\n"
        "fmla v17.4s, v5.4s, v6.s[0]\n"
        "fmla v19.4s, v5.4s, v6.s[1]\n"
        "fmla v21.4s, v5.4s, v6.s[2]\n"
        "fmla v23.4s, v5.4s, v6.s[3]\n"
        "ldr q5, [%[lhs_ptr], #-80]\n"
        "fmla v16.4s, v8.4s, v10.s[0]\n"
        "fmla v18.4s, v8.4s, v10.s[1]\n"
        "ldr q6, [%[rhs_ptr], #-96]\n"
        "fmla v20.4s, v8.4s, v10.s[2]\n"
        "fmla v22.4s, v8.4s, v10.s[3]\n"

        "fmla v24.4s, v8.4s, v11.s[0]\n"
        "fmla v26.4s, v8.4s, v11.s[1]\n"
        "fmla v28.4s, v8.4s, v11.s[2]\n"
        "fmla v30.4s, v8.4s, v11.s[3]\n"
        "ldr q8, [%[lhs_ptr], #-64]\n"
        "fmla v25.4s, v9.4s, v11.s[0]\n"
        "fmla v27.4s, v9.4s, v11.s[1]\n"
        "fmla v29.4s, v9.4s, v11.s[2]\n"
        "fmla v31.4s, v9.4s, v11.s[3]\n"
        "ldr q11, [%[rhs_ptr], #-48]\n"
        "fmla v17.4s, v9.4s, v10.s[0]\n"
        "fmla v19.4s, v9.4s, v10.s[1]\n"
        "fmla v21.4s, v9.4s, v10.s[2]\n"
        "fmla v23.4s, v9.4s, v10.s[3]\n"
        "ldr q9, [%[lhs_ptr], #-48]\n"
        "fmla v16.4s, v12.4s, v14.s[0]\n"
        "fmla v18.4s, v12.4s, v14.s[1]\n"
        "ldr q10, [%[rhs_ptr], #-64]\n"
        "fmla v20.4s, v12.4s, v14.s[2]\n"
        "fmla v22.4s, v12.4s, v14.s[3]\n"

        "fmla v24.4s, v12.4s, v15.s[0]\n"
        "fmla v26.4s, v12.4s, v15.s[1]\n"
        "fmla v28.4s, v12.4s, v15.s[2]\n"
        "fmla v30.4s, v12.4s, v15.s[3]\n"
        "ldr q12, [%[lhs_ptr], #-32]\n"
        "fmla v25.4s, v13.4s, v15.s[0]\n"
        "fmla v27.4s, v13.4s, v15.s[1]\n"
        "fmla v29.4s, v13.4s, v15.s[2]\n"
        "fmla v31.4s, v13.4s, v15.s[3]\n"
        "ldr q15, [%[rhs_ptr], #-16]\n"
        "fmla v17.4s, v13.4s, v14.s[0]\n"
        "fmla v19.4s, v13.4s, v14.s[1]\n"
        "fmla v21.4s, v13.4s, v14.s[2]\n"
        "fmla v23.4s, v13.4s, v14.s[3]\n"
        "ldr q13, [%[lhs_ptr], #-16]\n"
        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "ldr q14, [%[rhs_ptr], #-32]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        "add w1, w1, #4\n"
        "cmp w1, w2\n"
        "blt 80b\n"

        "fmla v16.4s, v4.4s, v6.s[0]\n"
        "fmla v18.4s, v4.4s, v6.s[1]\n"
        "fmla v20.4s, v4.4s, v6.s[2]\n"
        "fmla v22.4s, v4.4s, v6.s[3]\n"
        "fmla v24.4s, v4.4s, v7.s[0]\n"
        "fmla v26.4s, v4.4s, v7.s[1]\n"
        "fmla v28.4s, v4.4s, v7.s[2]\n"
        "fmla v30.4s, v4.4s, v7.s[3]\n"
        "fmla v25.4s, v5.4s, v7.s[0]\n"
        "fmla v27.4s, v5.4s, v7.s[1]\n"
        "fmla v29.4s, v5.4s, v7.s[2]\n"
        "fmla v31.4s, v5.4s, v7.s[3]\n"
        "fmla v17.4s, v5.4s, v6.s[0]\n"
        "fmla v19.4s, v5.4s, v6.s[1]\n"
        "fmla v21.4s, v5.4s, v6.s[2]\n"
        "fmla v23.4s, v5.4s, v6.s[3]\n"

        "fmla v16.4s, v8.4s, v10.s[0]\n"
        "fmla v18.4s, v8.4s, v10.s[1]\n"
        "fmla v20.4s, v8.4s, v10.s[2]\n"
        "fmla v22.4s, v8.4s, v10.s[3]\n"
        "fmla v24.4s, v8.4s, v11.s[0]\n"
        "fmla v26.4s, v8.4s, v11.s[1]\n"
        "fmla v28.4s, v8.4s, v11.s[2]\n"
        "fmla v30.4s, v8.4s, v11.s[3]\n"
        "fmla v25.4s, v9.4s, v11.s[0]\n"
        "fmla v27.4s, v9.4s, v11.s[1]\n"
        "fmla v29.4s, v9.4s, v11.s[2]\n"
        "fmla v31.4s, v9.4s, v11.s[3]\n"
        "fmla v17.4s, v9.4s, v10.s[0]\n"
        "fmla v19.4s, v9.4s, v10.s[1]\n"
        "fmla v21.4s, v9.4s, v10.s[2]\n"
        "fmla v23.4s, v9.4s, v10.s[3]\n"

        "fmla v16.4s, v12.4s, v14.s[0]\n"
        "fmla v18.4s, v12.4s, v14.s[1]\n"
        "fmla v20.4s, v12.4s, v14.s[2]\n"
        "fmla v22.4s, v12.4s, v14.s[3]\n"
        "fmla v24.4s, v12.4s, v15.s[0]\n"
        "fmla v26.4s, v12.4s, v15.s[1]\n"
        "fmla v28.4s, v12.4s, v15.s[2]\n"
        "fmla v30.4s, v12.4s, v15.s[3]\n"
        "fmla v25.4s, v13.4s, v15.s[0]\n"
        "fmla v27.4s, v13.4s, v15.s[1]\n"
        "fmla v29.4s, v13.4s, v15.s[2]\n"
        "fmla v31.4s, v13.4s, v15.s[3]\n"
        "fmla v17.4s, v13.4s, v14.s[0]\n"
        "fmla v19.4s, v13.4s, v14.s[1]\n"
        "fmla v21.4s, v13.4s, v14.s[2]\n"
        "fmla v23.4s, v13.4s, v14.s[3]\n"

        "78:\n"
#endif

        // Accumulation loop
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"
        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ld1 {v4.4s}, [%[rhs_ptr]], #16\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "add w1, w1, #1\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "cmp w1, w12\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "mov v2.16b, v4.16b\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "blt 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Determine the channel index.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition.
        // Jump based on channel dimension.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v8.4s, v14.s[0]\n"
        "dup v9.4s, v14.s[1]\n"
        "dup v10.4s, v14.s[2]\n"
        "dup v11.4s, v14.s[3]\n"
        "dup v12.4s, v15.s[0]\n"
        "dup v13.4s, v15.s[1]\n"
        "dup v14.4s, v15.s[2]\n"
        "dup v15.4s, v15.s[3]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "fadd v17.4s, v17.4s, v8.4s\n"
        "fadd v18.4s, v18.4s, v9.4s\n"
        "fadd v19.4s, v19.4s, v9.4s\n"
        "fadd v20.4s, v20.4s, v10.4s\n"
        "fadd v21.4s, v21.4s, v10.4s\n"
        "fadd v22.4s, v22.4s, v11.4s\n"
        "fadd v23.4s, v23.4s, v11.4s\n"
        "fadd v24.4s, v24.4s, v12.4s\n"
        "fadd v25.4s, v25.4s, v12.4s\n"
        "fadd v26.4s, v26.4s, v13.4s\n"
        "fadd v27.4s, v27.4s, v13.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v14.4s\n"
        "fadd v30.4s, v30.4s, v15.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "7:\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// A fork of the standard float kernel where we omit the manual loop unrolling
// to recover performance on the X1. For now, the X1 core is the only CPU that
// uses this kernel.
void KernelFloatNeonX1(const KernelParamsFloat<8, 8>& params) {
  CheckOffsetsInKernelParamsFloat(params);
  profiler::ScopeLabel label("Kernel (kNeon) X1");

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v15 are used to load data from LHS and RHS.
  // At least v0 and v1 are used to load a 8x1 block of LHS, and v2 and
  // v3 are used to load a 1x8 block of RHS, like this:
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------|
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------|
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for floading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        // Accumulation loop
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"
        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ld1 {v4.4s}, [%[rhs_ptr]], #16\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "add w1, w1, #1\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "cmp w1, w12\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "mov v2.16b, v4.16b\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "blt 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Determine the channel index.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition.
        // Jump based on channel dimension.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v8.4s, v14.s[0]\n"
        "dup v9.4s, v14.s[1]\n"
        "dup v10.4s, v14.s[2]\n"
        "dup v11.4s, v14.s[3]\n"
        "dup v12.4s, v15.s[0]\n"
        "dup v13.4s, v15.s[1]\n"
        "dup v14.4s, v15.s[2]\n"
        "dup v15.4s, v15.s[3]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "fadd v17.4s, v17.4s, v8.4s\n"
        "fadd v18.4s, v18.4s, v9.4s\n"
        "fadd v19.4s, v19.4s, v9.4s\n"
        "fadd v20.4s, v20.4s, v10.4s\n"
        "fadd v21.4s, v21.4s, v10.4s\n"
        "fadd v22.4s, v22.4s, v11.4s\n"
        "fadd v23.4s, v23.4s, v11.4s\n"
        "fadd v24.4s, v24.4s, v12.4s\n"
        "fadd v25.4s, v25.4s, v12.4s\n"
        "fadd v26.4s, v26.4s, v13.4s\n"
        "fadd v27.4s, v27.4s, v13.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v14.4s\n"
        "fadd v30.4s, v30.4s, v15.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "7:\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Variant of KernelFloatNeon tuned for in-order CPUs that do not
// support dotprod (while dotprod by itself is not relevant to floating-point,
// this additional bit of information that we have about the target happens to
// be useful here).
//
// So a typical target CPU here would be ARM Cortex-A53 or the original
// Cortex-A55.
//
// This kernel is similar to and inspired by gemmlowp's
// NEON_64bit_GEMM_Float32_WithScalar_A53.
// which was contributed by David Mansell with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A53:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4215
void KernelFloatNeonA55ish(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeon, optimized for in-order cores)");

  CheckOffsetsInKernelParamsFloat(params);

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v3 are used to load data from LHS and RHS.
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------|
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------|
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // There is no RUY_OPT_MAX_STREAMING 4x-unrolled part in this kernel because
  // we did not observe a benefit of such partial unrolling on in-order CPUs.
  //
  // v4 -- v7 are unused, and v8 -- v15 are used for floading parameters used
  // for the post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"


        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v17)
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v18)
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v19)
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v23)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v24)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v25)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v26)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v27)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "cmp w1, #0\n"
        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        // Accumulation loop
        "beq 79f\n"

        "2:\n"

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "ldr x2, [%[lhs_ptr], #8]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ldr x3, [%[lhs_ptr], #24]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "ldr x5, [%[rhs_ptr], #24]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ldr x4, [%[rhs_ptr], #8]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "subs w1, w1, #1\n"
        "ldr d0, [%[lhs_ptr]], #32\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ins v0.d[1], x2\n"
        "ldr d3, [%[rhs_ptr], #16]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "ins v3.d[1], x5\n"
        "ldr d4, [%[rhs_ptr]], #32\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "ins v4.d[1], x4\n"
        "ldr d1, [%[lhs_ptr], #-16]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "ins v1.d[1], x3\n"
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        "mov v2.16b, v4.16b\n"
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "bne 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Determine the channel index.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.

        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition.
        // Jump based on channel dimension.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v8.4s, v14.s[0]\n"
        "dup v9.4s, v14.s[1]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "dup v10.4s, v14.s[2]\n"
        "fadd v17.4s, v17.4s, v8.4s\n"
        "dup v11.4s, v14.s[3]\n"
        "fadd v18.4s, v18.4s, v9.4s\n"
        "dup v12.4s, v15.s[0]\n"
        "fadd v19.4s, v19.4s, v9.4s\n"
        "dup v13.4s, v15.s[1]\n"
        "fadd v20.4s, v20.4s, v10.4s\n"
        "dup v14.4s, v15.s[2]\n"
        "fadd v21.4s, v21.4s, v10.4s\n"
        "dup v15.4s, v15.s[3]\n"
        "fadd v22.4s, v22.4s, v11.4s\n"
        "fadd v23.4s, v23.4s, v11.4s\n"
        "fadd v24.4s, v24.4s, v12.4s\n"
        "fadd v25.4s, v25.4s, v12.4s\n"
        "fadd v26.4s, v26.4s, v13.4s\n"
        "fadd v27.4s, v27.4s, v13.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v14.4s\n"
        "fadd v30.4s, v30.4s, v15.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "7:\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Variant of KernelFloatNeonA55ish tuned for in-order CPUs that do
// support dotprod (while dotprod by itself is not relevant to floating-point,
// this additional bit of information that we have about the target happens to
// be useful here).
//
// So a typical target CPU here would be ARM Cortex-A55r1.
//
// This kernel is similar to and inspired by gemmlowp's
// NEON_64bit_GEMM_Float32_WithScalar_A55r1.
// which was contributed by David Mansell with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A55r1:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4412
void KernelFloatNeonDotprodA55ish(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label(
      "Kernel (kNeonDotprod, optimized for in-order cores)");

  CheckOffsetsInKernelParamsFloat(params);

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v3 are used to load data from LHS and RHS.
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------|
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------|
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // There is no RUY_OPT_MAX_STREAMING 4x-unrolled part in this kernel because
  // we did not observe a benefit of such partial unrolling on in-order CPUs.
  //
  // v4 -- v7 are unused, and v8 -- v15 are used for floading parameters used
  // for the post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "movi " #reg ".4s, #0\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"


        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v17)
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v18)
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v19)
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v23)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v24)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v25)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v26)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v27)
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "cmp w1, #0\n"
        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        // Accumulation loop
        "beq 79f\n"

        "2:\n"

        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "ldr x2, [%[lhs_ptr], #8]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ldr x3, [%[lhs_ptr], #24]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "ldr x5, [%[rhs_ptr], #24]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ldr d0, [%[lhs_ptr]], #32\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "ldr x4, [%[rhs_ptr], #8]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "subs w1, w1, #1\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "ins v0.d[1], x2\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr d3, [%[rhs_ptr], #16]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "ins v3.d[1], x5\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "ldr d4, [%[rhs_ptr]], #32\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "ins v4.d[1], x4\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        RUY_PREFETCH_LOAD("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "ldr d1, [%[lhs_ptr], #-16]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "ins v1.d[1], x3\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "mov v2.16b, v4.16b\n"
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "bne 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Determine the channel index.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "csel w3, %w[row], %w[col], eq\n"

        // Offset the bias pointer as needed given the current row, col.
        "add x5, x1, x3, lsl #2\n"

        // If there is no bias, use no offset, just address the passed zero
        // data.

        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition.
        // Jump based on channel dimension.
        "tst w4, #" RUY_STR(RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) "\n"
        "bne 6f\n"
        // Case where channels are rows
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "b 7f\n"

        "6:\n"
        // Case where channels are columns
        "dup v8.4s, v14.s[0]\n"
        "dup v9.4s, v14.s[1]\n"
        "fadd v16.4s, v16.4s, v8.4s\n"
        "dup v10.4s, v14.s[2]\n"
        "fadd v17.4s, v17.4s, v8.4s\n"
        "dup v11.4s, v14.s[3]\n"
        "fadd v18.4s, v18.4s, v9.4s\n"
        "dup v12.4s, v15.s[0]\n"
        "fadd v19.4s, v19.4s, v9.4s\n"
        "dup v13.4s, v15.s[1]\n"
        "fadd v20.4s, v20.4s, v10.4s\n"
        "dup v14.4s, v15.s[2]\n"
        "fadd v21.4s, v21.4s, v10.4s\n"
        "dup v15.4s, v15.s[3]\n"
        "fadd v22.4s, v22.4s, v11.4s\n"
        "fadd v23.4s, v23.4s, v11.4s\n"
        "fadd v24.4s, v24.4s, v12.4s\n"
        "fadd v25.4s, v25.4s, v12.4s\n"
        "fadd v26.4s, v26.4s, v13.4s\n"
        "fadd v27.4s, v27.4s, v13.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v14.4s\n"
        "fadd v30.4s, v30.4s, v15.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"
        "7:\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x3]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        RUY_PREFETCH_STORE("prfm pstl1strm, [x4]\n")
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}
#undef RUY_OFFSET_BIAS
#undef RUY_OFFSET_FLAGS
#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR

#endif  // RUY_PLATFORM_NEON_64 && RUY_OPT(ASM) && !defined(_MSC_VER)

}  // namespace ruy

// MSVC ARM64 NEON intrinsic kernel/pack implementations for ruy.
//
// These replace the GAS-asm kernels in kernel_arm64.cc and pack_arm.cc, which
// are guarded with !defined(_MSC_VER) / !(_MSC_VER && _M_ARM64).
//
// Data layout notes (packed matrices):
//
// Float kernel  (FixedKernelLayout<kRowMajor, 1, 8>):
//   lhs_base_ptr = lhs.data + start_row * lhs.layout.stride  (floats)
//   lhs_stride   = sizeof(float) * lhs.layout.stride  (bytes between tiles/8)
//   Per depth step d: 8 floats at lhs_ptr[d*8 .. d*8+7]  (advance +8 floats)
//   Row-tile advance: lhs_ptr += lhs_stride/sizeof(float) * 8  (8 x depth)
//
// Int8 dotprod  (FixedKernelLayout<kColMajor, 4, 8>):
//   lhs_base_ptr = lhs.data + start_row * lhs.layout.stride  (int8 units)
//   lhs_stride   = lhs.layout.stride  (int8 units = depth, NOT bytes)
//   Per 4-depth step: 32 bytes = 2 x int8x16 (advance +32 bytes in ptr)
//   Row-tile advance: lhs_ptr += lhs_stride * 8  (in int8 units)
//
// dst_base_ptr = dst.data + start_col * dst.layout.stride + start_row (elems)
//   dst_stride = sizeof(DstScalar) * dst.layout.stride  (bytes per column)

#if defined(_MSC_VER) && defined(_M_ARM64)

#include <algorithm>
#include <cstring>

#define NOMINMAX  // Prevent windows.h from defining min/max macros.
#include <arm_neon.h>
#include <windows.h>

#include "ruy/apply_multiplier.h"

namespace ruy {

// Transpose 4 int8x16 vectors and store at stride 32.
// After transpose, output[4k..4k+15] = {col0[k*4..k*4+3], col1[k*4..k*4+3],
//                                        col2[k*4..k*4+3], col3[k*4..k*4+3]}
// for k = 0,1,2,3 (corresponding to depths 0..3, 4..7, 8..11, 12..15).
// Strides (GAS uses {0, 32, 64, 96}); we store 4 groups out of 4.
#define RUY_DOTPROD_PACK_TRANSPOSE_STORE(v0_, v1_, v2_, v3_, p_, do_sums)    \
    do {                                                                      \
        /* XOR for uint8→int8 conversion */                                   \
        uint8x16_t e0_ = veorq_u8((v0_), xorv);                              \
        uint8x16_t e1_ = veorq_u8((v1_), xorv);                              \
        uint8x16_t e2_ = veorq_u8((v2_), xorv);                              \
        uint8x16_t e3_ = veorq_u8((v3_), xorv);                              \
        /* Transpose: trn1/trn2 at int32 granularity */                       \
        /* Then combine halves to get 4-depth groups */                        \
        int32x4_t i0_ = vreinterpretq_s32_u8(e0_);                           \
        int32x4_t i1_ = vreinterpretq_s32_u8(e1_);                           \
        int32x4_t i2_ = vreinterpretq_s32_u8(e2_);                           \
        int32x4_t i3_ = vreinterpretq_s32_u8(e3_);                           \
        int32x4x2_t t01_ = vtrnq_s32(i0_, i1_);                              \
        int32x4x2_t t23_ = vtrnq_s32(i2_, i3_);                              \
        /* Group 0 (depths 0..3): low halves */                               \
        int8x16_t g0_ = vreinterpretq_s8_s64(vcombine_s64(                   \
            vget_low_s64(vreinterpretq_s64_s32(t01_.val[0])),                 \
            vget_low_s64(vreinterpretq_s64_s32(t23_.val[0]))));               \
        /* Group 1 (depths 4..7): low halves of t01_[1] and t23_[1] */       \
        int8x16_t g1_ = vreinterpretq_s8_s64(vcombine_s64(                   \
            vget_low_s64(vreinterpretq_s64_s32(t01_.val[1])),                 \
            vget_low_s64(vreinterpretq_s64_s32(t23_.val[1]))));               \
        /* Group 2 (depths 8..11): high halves of t01_[0] and t23_[0] */     \
        int8x16_t g2_ = vreinterpretq_s8_s64(vcombine_s64(                   \
            vget_high_s64(vreinterpretq_s64_s32(t01_.val[0])),                \
            vget_high_s64(vreinterpretq_s64_s32(t23_.val[0]))));              \
        /* Group 3 (depths 12..15): high halves of t01_[1] and t23_[1] */    \
        int8x16_t g3_ = vreinterpretq_s8_s64(vcombine_s64(                   \
            vget_high_s64(vreinterpretq_s64_s32(t01_.val[1])),                \
            vget_high_s64(vreinterpretq_s64_s32(t23_.val[1]))));              \
        /* Accumulate per-row sums using sdot-with-ones on TRANSPOSED groups. */ \
        /* sum_rows[j] += sum(g_k[4j..4j+3]) for each group k processed.     */ \
        /* This matches GAS: sdot v28.4s, v20.16b, v27.16b (v27=ones).       */ \
        if (do_sums) {                                                        \
            sum_rows = vdotq_s32(sum_rows, g0_, ones);                        \
            sum_rows = vdotq_s32(sum_rows, g1_, ones);                        \
            sum_rows = vdotq_s32(sum_rows, g2_, ones);                        \
            sum_rows = vdotq_s32(sum_rows, g3_, ones);                        \
        }                                                                     \
        vst1q_s8((p_),       g0_);                                            \
        vst1q_s8((p_) + 32,  g1_);                                            \
        vst1q_s8((p_) + 64,  g2_);                                            \
        vst1q_s8((p_) + 96,  g3_);                                            \
        (p_) += 128;  /* advance past 4 groups × 32-byte stride */            \
    } while (0)

// Transpose a 4x4 float block and store each row at stride 8 floats (32 bytes).
// v0..v3 are the 4 columns (each = 4 depth values).
// row k = {col0[k], col1[k], col2[k], col3[k]} is stored at p + k*8.
// Uses vtrnq_f32 (available in MSVC ARM NEON) which returns float32x4x2_t.
// Note: no lambda to avoid MSVC lambda capture of void* issues.
#define RUY_PACK_FLOAT_TRANSPOSE_STORE(v0_, v1_, v2_, v3_, p_)           \
    do {                                                                  \
        float32x4x2_t trn01_ = vtrnq_f32((v0_), (v1_));                  \
        float32x4x2_t trn23_ = vtrnq_f32((v2_), (v3_));                  \
        /* row0 = {v0[0],v1[0],v2[0],v3[0]} */                           \
        float32x4_t r0_ = vcombine_f32(vget_low_f32(trn01_.val[0]),       \
                                       vget_low_f32(trn23_.val[0]));      \
        float32x4_t r1_ = vcombine_f32(vget_low_f32(trn01_.val[1]),       \
                                       vget_low_f32(trn23_.val[1]));      \
        float32x4_t r2_ = vcombine_f32(vget_high_f32(trn01_.val[0]),      \
                                       vget_high_f32(trn23_.val[0]));     \
        float32x4_t r3_ = vcombine_f32(vget_high_f32(trn01_.val[1]),      \
                                       vget_high_f32(trn23_.val[1]));     \
        vst1q_f32((p_),      r0_);                                        \
        vst1q_f32((p_) + 8,  r1_);                                        \
        vst1q_f32((p_) + 16, r2_);                                        \
        vst1q_f32((p_) + 24, r3_);                                        \
        (p_) += 32;                                                       \
    } while (0)

// Returns true if the CPU supports the ARM v8.2 dot-product extension.
static bool HasDotprod() {
  static const bool cached =
      IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != FALSE;
  return cached;
}

// Applies (fixedpoint, exponent) multiplier per element in a 4-wide int32
// vector.  Calls MultiplyByQuantizedMultiplier element-by-element for
// bit-exact match with the reference and the GAS asm paths.
static inline int32x4_t ApplyMultiplierVec4(int32x4_t v, int32x4_t fp,
                                             int32x4_t ep) {
  // Use the reference implementation element-by-element to guarantee bit-exact
  // match with MultiplyByQuantizedMultiplier (which is the reference for ARM).
  // This is what all our ARM code paths must match (tolerated_max_diff = 0).
  int32_t v0[4], fp0[4], ep0[4], out[4];
  vst1q_s32(v0, v);
  vst1q_s32(fp0, fp);
  vst1q_s32(ep0, ep);
  for (int i = 0; i < 4; ++i)
    out[i] = detail::MultiplyByQuantizedMultiplier(v0[i], fp0[i], ep0[i]);
  return vld1q_s32(out);
}

// KernelFloatNeon: 8x8 float32 GEMM tile using NEON FMA intrinsics.
static void KernelFloatNeonImpl(const KernelParamsFloat<8, 8>& params) {
  // lhs_stride and rhs_stride are in bytes.
  const int lhs_stride_bytes = params.lhs_stride;
  const int rhs_stride_bytes = params.rhs_stride;
  const int dst_stride_elems =
      params.dst_stride / static_cast<int>(sizeof(float));

  // lhs_tile_stride: advance in float elements to go from one 8-row tile to
  // the next.  lhs_stride_bytes/sizeof(float) = depth, so:
  //   row tile r   starts at lhs.data + r * depth
  //   row tile r+8 starts at lhs.data + (r+8) * depth
  //   delta = 8 * depth = 8 * (lhs_stride_bytes / sizeof(float))
  const int lhs_tile_stride_elems =
      8 * (lhs_stride_bytes / static_cast<int>(sizeof(float)));
  const int rhs_tile_stride_elems =
      8 * (rhs_stride_bytes / static_cast<int>(sizeof(float)));

  const float* lhs_col_ptr = params.lhs_base_ptr;  // start_row tile
  const float* rhs_col_base = params.rhs_base_ptr;  // start_col tile
  float* dst_base = params.dst_base_ptr;

  for (int row = params.start_row; row <= params.last_row; row += 8) {
    const float* rhs_col_ptr = rhs_col_base;
    // dst_base_ptr = dst.data + start_col * dst_stride + start_row (elements)
    // For the current row tile, offset from start_row is (row - start_row).
    float* dst_row_ptr = dst_base + (row - params.start_row);

    for (int col = params.start_col; col <= params.last_col; col += 8) {
      float32x4_t lo[8], hi[8];
      for (int c = 0; c < 8; ++c) {
        lo[c] = vdupq_n_f32(0.f);
        hi[c] = vdupq_n_f32(0.f);
      }

      // Inner loop: one depth step = 8 floats per operand (FixedKernelLayout row=1, cols=8).
      const float* ld = lhs_col_ptr;
      const float* rd = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d, ld += 8, rd += 8) {
        float32x4_t l0 = vld1q_f32(ld);
        float32x4_t l1 = vld1q_f32(ld + 4);
        float32x4_t r0 = vld1q_f32(rd);
        float32x4_t r1 = vld1q_f32(rd + 4);
        lo[0] = vfmaq_laneq_f32(lo[0], l0, r0, 0);
        hi[0] = vfmaq_laneq_f32(hi[0], l1, r0, 0);
        lo[1] = vfmaq_laneq_f32(lo[1], l0, r0, 1);
        hi[1] = vfmaq_laneq_f32(hi[1], l1, r0, 1);
        lo[2] = vfmaq_laneq_f32(lo[2], l0, r0, 2);
        hi[2] = vfmaq_laneq_f32(hi[2], l1, r0, 2);
        lo[3] = vfmaq_laneq_f32(lo[3], l0, r0, 3);
        hi[3] = vfmaq_laneq_f32(hi[3], l1, r0, 3);
        lo[4] = vfmaq_laneq_f32(lo[4], l0, r1, 0);
        hi[4] = vfmaq_laneq_f32(hi[4], l1, r1, 0);
        lo[5] = vfmaq_laneq_f32(lo[5], l0, r1, 1);
        hi[5] = vfmaq_laneq_f32(hi[5], l1, r1, 1);
        lo[6] = vfmaq_laneq_f32(lo[6], l0, r1, 2);
        hi[6] = vfmaq_laneq_f32(hi[6], l1, r1, 2);
        lo[7] = vfmaq_laneq_f32(lo[7], l0, r1, 3);
        hi[7] = vfmaq_laneq_f32(hi[7], l1, r1, 3);
      }

      // Post-accumulation: optional bias, then clamp.
      const std::uint8_t flags = params.flags;
      bool ch_col = (flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) != 0;
      if (flags & RUY_ASM_FLAG_HAS_BIAS) {
        if (ch_col) {
          for (int c = 0; c < 8; ++c) {
            float32x4_t b = vdupq_n_f32(params.bias[col + c]);
            lo[c] = vaddq_f32(lo[c], b);
            hi[c] = vaddq_f32(hi[c], b);
          }
        } else {
          float32x4_t b0 = vld1q_f32(params.bias + row);
          float32x4_t b1 = vld1q_f32(params.bias + row + 4);
          for (int c = 0; c < 8; ++c) {
            lo[c] = vaddq_f32(lo[c], b0);
            hi[c] = vaddq_f32(hi[c], b1);
          }
        }
      }
      float32x4_t cmin = vdupq_n_f32(params.clamp_min);
      float32x4_t cmax = vdupq_n_f32(params.clamp_max);
      for (int c = 0; c < 8; ++c) {
        lo[c] = vmaxq_f32(vminq_f32(lo[c], cmax), cmin);
        hi[c] = vmaxq_f32(vminq_f32(hi[c], cmax), cmin);
      }

      // Store to dst (column-major).
      // dst_base_ptr = dst.data + start_col * dst_stride + start_row
      // For col offset c_abs = col - start_col (= 0, 8, 16, ...):
      //   dst for col c_abs = dst_base + c_abs * dst_stride_elems
      // For row offset r_abs = row - start_row:
      //   already included in dst_row_ptr
      int c_abs = col - params.start_col;
      float* dst_col_ptr = dst_row_ptr + c_abs * dst_stride_elems;

      int fit_r = std::min(params.dst_rows - row, 8);
      int fit_c = std::min(params.dst_cols - col, 8);
      bool full = (fit_r == 8) && (fit_c == 8);
      float tmp[8 * 8];
      float* wb = full ? dst_col_ptr : tmp;
      int ws = full ? dst_stride_elems : 8;
      for (int c = 0; c < 8; ++c) {
        vst1q_f32(wb + c * ws,     lo[c]);
        vst1q_f32(wb + c * ws + 4, hi[c]);
      }
      if (!full) {
        for (int c = 0; c < fit_c; ++c)
          std::memcpy(dst_col_ptr + c * dst_stride_elems, tmp + c * 8,
                      fit_r * sizeof(float));
      }

      rhs_col_ptr += rhs_tile_stride_elems;
    }  // col

    lhs_col_ptr += lhs_tile_stride_elems;
  }  // row
}

void KernelFloatNeon(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeon, MSVC NEON f32)");
  KernelFloatNeonImpl(params);
}
void KernelFloatNeonX1(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonX1, MSVC NEON f32)");
  KernelFloatNeonImpl(params);
}
void KernelFloatNeonA55ish(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonA55ish, MSVC NEON f32)");
  KernelFloatNeonImpl(params);
}
void KernelFloatNeonDotprodA55ish(const KernelParamsFloat<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprodA55ish, MSVC NEON f32)");
  KernelFloatNeonImpl(params);
}

// Kernel8bitNeonDotprod: 8x8 int8 GEMM tile using vdotq_laneq_s32.
// vdotq_laneq_s32(acc, a, b, lane) computes acc[i] += dot4(a[4i..], b[4*lane..])
// for each i, accumulating 4 int8 products into each int32 lane per call.
static void Kernel8bitNeonDotprodImpl(const KernelParams8bit<8, 8>& params) {
  const int depth = params.depth;
  RUY_DCHECK_EQ(depth % 4, 0);

  // lhs_stride is in int8 units (= depth).
  // Row-tile advance = 8 * lhs_stride int8 units.
  // Per 4-depth step: advance 32 bytes = 32 int8 units.
  const int lhs_tile_stride = 8 * params.lhs_stride;  // int8 units per row tile
  const int rhs_tile_stride = 8 * params.rhs_stride;  // bytes per col tile
  const int dst_stride_bytes = params.dst_stride;

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_base =
      static_cast<const std::int8_t*>(params.rhs_base_ptr);

  for (int row = params.start_row; row <= params.last_row; row += 8) {
    const std::int8_t* rhs_col_ptr = rhs_col_base;

    for (int col = params.start_col; col <= params.last_col; col += 8) {
      int32x4_t lo[8], hi[8];
      for (int c = 0; c < 8; ++c) {
        lo[c] = vdupq_n_s32(0);
        hi[c] = vdupq_n_s32(0);
      }

      // Inner loop: 4 depth elements per step, 32 bytes each side.
      const std::int8_t* ld = lhs_col_ptr;
      const std::int8_t* rd = rhs_col_ptr;
      for (int d = 0; d < depth; d += 4, ld += 32, rd += 32) {
        int8x16_t l0 = vld1q_s8(ld);       // rows 0..3, depths d..d+3
        int8x16_t l1 = vld1q_s8(ld + 16);  // rows 4..7, depths d..d+3
        int8x16_t r0 = vld1q_s8(rd);       // cols 0..3, depths d..d+3
        int8x16_t r1 = vld1q_s8(rd + 16);  // cols 4..7, depths d..d+3

        lo[0] = vdotq_laneq_s32(lo[0], l0, r0, 0);
        hi[0] = vdotq_laneq_s32(hi[0], l1, r0, 0);
        lo[1] = vdotq_laneq_s32(lo[1], l0, r0, 1);
        hi[1] = vdotq_laneq_s32(hi[1], l1, r0, 1);
        lo[2] = vdotq_laneq_s32(lo[2], l0, r0, 2);
        hi[2] = vdotq_laneq_s32(hi[2], l1, r0, 2);
        lo[3] = vdotq_laneq_s32(lo[3], l0, r0, 3);
        hi[3] = vdotq_laneq_s32(hi[3], l1, r0, 3);
        lo[4] = vdotq_laneq_s32(lo[4], l0, r1, 0);
        hi[4] = vdotq_laneq_s32(hi[4], l1, r1, 0);
        lo[5] = vdotq_laneq_s32(lo[5], l0, r1, 1);
        hi[5] = vdotq_laneq_s32(hi[5], l1, r1, 1);
        lo[6] = vdotq_laneq_s32(lo[6], l0, r1, 2);
        hi[6] = vdotq_laneq_s32(hi[6], l1, r1, 2);
        lo[7] = vdotq_laneq_s32(lo[7], l0, r1, 3);
        hi[7] = vdotq_laneq_s32(hi[7], l1, r1, 3);
      }

      // Post-accumulation.
      const std::uint8_t flags = params.flags;
      bool ch_col = (flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) != 0;

      // Step 1: add prod_zp_depth, add bias.
      {
        int32x4_t pzd = vdupq_n_s32(params.prod_zp_depth);
        if (flags & RUY_ASM_FLAG_HAS_BIAS) {
          if (ch_col) {
            for (int c = 0; c < 8; ++c) {
              int32x4_t b = vdupq_n_s32(params.bias[col + c]);
              lo[c] = vaddq_s32(vaddq_s32(lo[c], pzd), b);
              hi[c] = vaddq_s32(vaddq_s32(hi[c], pzd), b);
            }
          } else {
            int32x4_t b0 = vld1q_s32(params.bias + row);
            int32x4_t b1 = vld1q_s32(params.bias + row + 4);
            for (int c = 0; c < 8; ++c) {
              lo[c] = vaddq_s32(vaddq_s32(lo[c], pzd), b0);
              hi[c] = vaddq_s32(vaddq_s32(hi[c], pzd), b1);
            }
          }
        } else {
          for (int c = 0; c < 8; ++c) {
            lo[c] = vaddq_s32(lo[c], pzd);
            hi[c] = vaddq_s32(hi[c], pzd);
          }
        }
      }

      // Step 2: subtract lhs_zero_point * rhs_sums[col+c]
      if (flags & RUY_ASM_FLAG_HAS_RHS_SUMS) {
        int32x4_t lzp = vdupq_n_s32(params.lhs_zero_point);
        for (int c = 0; c < 8; ++c) {
          int32x4_t sub =
              vmulq_s32(lzp, vdupq_n_s32(params.rhs_sums[col + c]));
          lo[c] = vsubq_s32(lo[c], sub);
          hi[c] = vsubq_s32(hi[c], sub);
        }
      }

      // Step 3: subtract rhs_zero_point * lhs_sums[row..row+7]
      if (flags & RUY_ASM_FLAG_HAS_LHS_SUMS) {
        int32x4_t rzp = vdupq_n_s32(params.rhs_zero_point);
        int32x4_t ls0 = vmulq_s32(rzp, vld1q_s32(params.lhs_sums + row));
        int32x4_t ls1 = vmulq_s32(rzp, vld1q_s32(params.lhs_sums + row + 4));
        for (int c = 0; c < 8; ++c) {
          lo[c] = vsubq_s32(lo[c], ls0);
          hi[c] = vsubq_s32(hi[c], ls1);
        }
      }

      // Compute tile dst pointer.
      // dst_base_ptr = dst.data + start_col * dst_stride + start_row (in elems)
      // Tile at (row, col): col offset = (col - start_col) * dst_stride_bytes
      //                     row offset = (row - start_row) * elem_size
      int r_off = row - params.start_row;
      int c_off = col - params.start_col;

      int elem_size;
      switch (params.dst_type_id) {
        case RUY_ASM_TYPE_ID_INT32:  elem_size = 4; break;
        case RUY_ASM_TYPE_ID_INT16:  elem_size = 2; break;
        default:                     elem_size = 1; break;
      }
      std::uint8_t* tile_dst =
          static_cast<std::uint8_t*>(params.dst_base_ptr) +
          static_cast<std::ptrdiff_t>(c_off) * dst_stride_bytes +
          static_cast<std::ptrdiff_t>(r_off) * elem_size;

      int fit_r = std::min(params.dst_rows - row, 8);
      int fit_c = std::min(params.dst_cols - col, 8);
      bool full = (fit_r == 8) && (fit_c == 8);

      if (params.dst_type_id == RUY_ASM_TYPE_ID_INT32) {
        // No multiplier -- store int32 directly.
        std::int32_t* dst32 = reinterpret_cast<std::int32_t*>(tile_dst);
        int stride32 = dst_stride_bytes / sizeof(std::int32_t);
        std::int32_t tmp[8 * 8];
        std::int32_t* wb = full ? dst32 : tmp;
        int ws = full ? stride32 : 8;
        for (int c = 0; c < 8; ++c) {
          vst1q_s32(wb + c * ws,     lo[c]);
          vst1q_s32(wb + c * ws + 4, hi[c]);
        }
        if (!full) {
          for (int c = 0; c < fit_c; ++c)
            std::memcpy(dst32 + c * stride32, tmp + c * 8,
                        fit_r * sizeof(std::int32_t));
        }
        rhs_col_ptr += rhs_tile_stride;
        continue;
      }

      // Step 4: apply quantized multiplier.
      bool is_perchannel = (flags & RUY_ASM_FLAG_HAS_PERCHANNEL) != 0;
      if (ch_col) {
        int off = is_perchannel ? col : 0;
        for (int c = 0; c < 8; ++c) {
          int32x4_t fp = vdupq_n_s32(params.multiplier_fixedpoint[off + c]);
          int32x4_t ep = vdupq_n_s32(params.multiplier_exponent[off + c]);
          lo[c] = ApplyMultiplierVec4(lo[c], fp, ep);
          hi[c] = ApplyMultiplierVec4(hi[c], fp, ep);
        }
      } else {
        int off = is_perchannel ? row : 0;
        int32x4_t fp0 = vld1q_s32(params.multiplier_fixedpoint + off);
        int32x4_t fp1 = vld1q_s32(params.multiplier_fixedpoint + off + 4);
        int32x4_t ep0 = vld1q_s32(params.multiplier_exponent + off);
        int32x4_t ep1 = vld1q_s32(params.multiplier_exponent + off + 4);
        for (int c = 0; c < 8; ++c) {
          lo[c] = ApplyMultiplierVec4(lo[c], fp0, ep0);
          hi[c] = ApplyMultiplierVec4(hi[c], fp1, ep1);
        }
      }

      // Step 5: add dst_zero_point.
      {
        int32x4_t dzp = vdupq_n_s32(params.dst_zero_point);
        for (int c = 0; c < 8; ++c) {
          lo[c] = vaddq_s32(lo[c], dzp);
          hi[c] = vaddq_s32(hi[c], dzp);
        }
      }

      // Step 6: clamp and quantize to output type.
      if (params.dst_type_id == RUY_ASM_TYPE_ID_INT16) {
        std::int16_t* dst16 = reinterpret_cast<std::int16_t*>(tile_dst);
        int stride16 = dst_stride_bytes / sizeof(std::int16_t);
        int32x4_t cmin32 = vdupq_n_s32(params.clamp_min);
        int32x4_t cmax32 = vdupq_n_s32(params.clamp_max);
        std::int16_t tmp[8 * 8];
        std::int16_t* wb = full ? dst16 : tmp;
        int ws = full ? stride16 : 8;
        for (int c = 0; c < 8; ++c) {
          int32x4_t cl0 = vmaxq_s32(vminq_s32(lo[c], cmax32), cmin32);
          int32x4_t cl1 = vmaxq_s32(vminq_s32(hi[c], cmax32), cmin32);
          int16x8_t s = vcombine_s16(vqmovn_s32(cl0), vqmovn_s32(cl1));
          vst1q_s16(wb + c * ws, s);
        }
        if (!full) {
          for (int c = 0; c < fit_c; ++c)
            std::memcpy(dst16 + c * stride16, tmp + c * 8,
                        fit_r * sizeof(std::int16_t));
        }
      } else {
        // uint8 or int8 output.
        bool is_s8 = (params.dst_type_id == RUY_ASM_TYPE_ID_INT8);
        std::uint8_t tmp[8 * 8];
        std::uint8_t* dst8 = tile_dst;
        std::uint8_t* wb = full ? dst8 : tmp;
        int ws = full ? dst_stride_bytes : 8;
        auto cmin8 = static_cast<std::int8_t>(params.clamp_min);
        auto cmax8 = static_cast<std::int8_t>(params.clamp_max);
        auto ucmin8 = static_cast<std::uint8_t>(params.clamp_min);
        auto ucmax8 = static_cast<std::uint8_t>(params.clamp_max);
        for (int c = 0; c < 8; ++c) {
          int16x8_t s = vcombine_s16(vqmovn_s32(lo[c]), vqmovn_s32(hi[c]));
          if (is_s8) {
            int8x8_t b = vqmovn_s16(s);
            b = vmax_s8(b, vdup_n_s8(cmin8));
            b = vmin_s8(b, vdup_n_s8(cmax8));
            vst1_s8(reinterpret_cast<std::int8_t*>(wb + c * ws), b);
          } else {
            uint8x8_t b = vqmovun_s16(s);
            b = vmax_u8(b, vdup_n_u8(ucmin8));
            b = vmin_u8(b, vdup_n_u8(ucmax8));
            vst1_u8(wb + c * ws, b);
          }
        }
        if (!full) {
          for (int c = 0; c < fit_c; ++c)
            std::memcpy(dst8 + c * dst_stride_bytes, tmp + c * 8, fit_r);
        }
      }

      rhs_col_ptr += rhs_tile_stride;
    }  // col

    lhs_col_ptr += lhs_tile_stride;
  }  // row
}

void Kernel8bitNeonDotprod(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprod, MSVC NEON)");
  if (!HasDotprod()) return;
  Kernel8bitNeonDotprodImpl(params);
}
void Kernel8bitNeonDotprodX1(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprodX1, MSVC NEON)");
  if (!HasDotprod()) return;
  Kernel8bitNeonDotprodImpl(params);
}
void Kernel8bitNeonDotprodA55ish(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprodA55ish, MSVC NEON)");
  if (!HasDotprod()) return;
  Kernel8bitNeonDotprodImpl(params);
}
void Kernel8bitNeonDotprod1Col(const KernelParams8bit<8, 8>& params) {
  profiler::ScopeLabel label("Kernel (kNeonDotprod1Col, MSVC NEON)");
  if (!HasDotprod()) return;
  Kernel8bitNeonDotprodImpl(params);
}

// Kernel8bitNeon: 4x4 int8 GEMM tile without dotprod.
// Uses vmull_s8 + vmlal_high_s8 + vpadalq_s16, mirroring the GAS kernel.
//
// Used when DotProd is not available.  NEON intrinsic inner loop using
// vmull_s8 / vmlal_high_s8 / vpadalq_s16 — mirrors the GAS kernel exactly.

// Mixed-precision kernel: i8×i16→i16 and i16×i8→i16, tile 4×4, depth step 8.
// Inner loop widens the int8 operand via vmovl_s8, then uses vmull_s16 +
// vmlal_high_s16 for 8 int16 products per step into a 4-wide int32 accumulator.

// Multiply-accumulate: 8 int16 products (low 4 via vmull_s16, high 4 via
// vmlal_high_s16) folded into a 4-wide int32 accumulator.
#define RUY_MIX_MAC(acc_, ln_, rn_)                                    \
    do {                                                                \
        int32x4_t p_ = vmull_s16(vget_low_s16(ln_),                    \
                                 vget_low_s16(rn_));                    \
        p_ = vmlal_high_s16(p_, ln_, rn_);                             \
        acc_ = vaddq_s32(acc_, p_);                                     \
    } while (0)

// lhs_is_int8 = true  → LHS packed as int8  (i8×i16 case)
// lhs_is_int8 = false → LHS packed as int16 (i16×i8 case)
template <bool lhs_is_int8>
static void Kernel8bitNeonMixedImpl(const KernelParams8bit<4, 4>& params) {
  const int depth = params.depth;
  const int dst_stride_bytes = params.dst_stride;

  // For int8 LHS: lhs_base_ptr is const int8*,  stride in int8 elements.
  // For int16 LHS: lhs_base_ptr reinterprets int16* as int8*; stride is in
  //   int16 elements — multiply by sizeof(int16_t) to get bytes.
  const int lhs_elem_bytes = lhs_is_int8 ? 1 : 2;

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_base =
      static_cast<const std::int8_t*>(params.rhs_base_ptr);

  for (int row = params.start_row; row <= params.last_row; row += 4) {
    const std::int8_t* rhs_col_ptr = rhs_col_base;
    for (int col = params.start_col; col <= params.last_col; col += 4) {
      // 16 int32 accumulators (4 rows × 4 cols), each a 4-element partial sum.
      int32x4_t acc00=vdupq_n_s32(0), acc01=vdupq_n_s32(0);
      int32x4_t acc02=vdupq_n_s32(0), acc03=vdupq_n_s32(0);
      int32x4_t acc10=vdupq_n_s32(0), acc11=vdupq_n_s32(0);
      int32x4_t acc12=vdupq_n_s32(0), acc13=vdupq_n_s32(0);
      int32x4_t acc20=vdupq_n_s32(0), acc21=vdupq_n_s32(0);
      int32x4_t acc22=vdupq_n_s32(0), acc23=vdupq_n_s32(0);
      int32x4_t acc30=vdupq_n_s32(0), acc31=vdupq_n_s32(0);
      int32x4_t acc32=vdupq_n_s32(0), acc33=vdupq_n_s32(0);

      const std::int8_t* lhs_ptr = lhs_col_ptr;
      const std::int8_t* rhs_ptr = rhs_col_ptr;

      // Depth loop: 8 elements per step (each col = 8 int16 = 16 bytes).
      for (int d = 0; d < depth; d += 8) {
        int16x8_t l0, l1, l2, l3;  // LHS rows 0..3, 8 int16 values each
        int16x8_t r0, r1, r2, r3;  // RHS cols 0..3

        if (lhs_is_int8) {
          // LHS: load 8 int8, sign-extend to int16.
          // Packed layout: 4 rows × 8 int8 = 32 bytes per depth block.
          int8x8_t lb0 = vld1_s8(lhs_ptr);      lhs_ptr += 8;
          int8x8_t lb1 = vld1_s8(lhs_ptr);      lhs_ptr += 8;
          int8x8_t lb2 = vld1_s8(lhs_ptr);      lhs_ptr += 8;
          int8x8_t lb3 = vld1_s8(lhs_ptr);      lhs_ptr += 8;
          l0 = vmovl_s8(lb0);
          l1 = vmovl_s8(lb1);
          l2 = vmovl_s8(lb2);
          l3 = vmovl_s8(lb3);
          // RHS: load 8 int16 per col (4 cols × 16 bytes = 64 bytes).
          r0 = vld1q_s16(reinterpret_cast<const std::int16_t*>(rhs_ptr));
          rhs_ptr += 16;
          r1 = vld1q_s16(reinterpret_cast<const std::int16_t*>(rhs_ptr));
          rhs_ptr += 16;
          r2 = vld1q_s16(reinterpret_cast<const std::int16_t*>(rhs_ptr));
          rhs_ptr += 16;
          r3 = vld1q_s16(reinterpret_cast<const std::int16_t*>(rhs_ptr));
          rhs_ptr += 16;
        } else {
          // LHS: load 8 int16 per row (4 rows × 16 bytes = 64 bytes).
          l0 = vld1q_s16(reinterpret_cast<const std::int16_t*>(lhs_ptr));
          lhs_ptr += 16;
          l1 = vld1q_s16(reinterpret_cast<const std::int16_t*>(lhs_ptr));
          lhs_ptr += 16;
          l2 = vld1q_s16(reinterpret_cast<const std::int16_t*>(lhs_ptr));
          lhs_ptr += 16;
          l3 = vld1q_s16(reinterpret_cast<const std::int16_t*>(lhs_ptr));
          lhs_ptr += 16;
          // RHS: load 8 int8, sign-extend to int16.
          int8x8_t rb0 = vld1_s8(rhs_ptr);      rhs_ptr += 8;
          int8x8_t rb1 = vld1_s8(rhs_ptr);      rhs_ptr += 8;
          int8x8_t rb2 = vld1_s8(rhs_ptr);      rhs_ptr += 8;
          int8x8_t rb3 = vld1_s8(rhs_ptr);      rhs_ptr += 8;
          r0 = vmovl_s8(rb0);
          r1 = vmovl_s8(rb1);
          r2 = vmovl_s8(rb2);
          r3 = vmovl_s8(rb3);
        }

        // MAC: vmull_s16 (low 4) + vmlal_high_s16 (high 4) = 8 products → int32.
        // Each acc_rc accumulates 4 horizontal partial sums.
        RUY_MIX_MAC(acc00, l0, r0); RUY_MIX_MAC(acc01, l0, r1);
        RUY_MIX_MAC(acc02, l0, r2); RUY_MIX_MAC(acc03, l0, r3);
        RUY_MIX_MAC(acc10, l1, r0); RUY_MIX_MAC(acc11, l1, r1);
        RUY_MIX_MAC(acc12, l1, r2); RUY_MIX_MAC(acc13, l1, r3);
        RUY_MIX_MAC(acc20, l2, r0); RUY_MIX_MAC(acc21, l2, r1);
        RUY_MIX_MAC(acc22, l2, r2); RUY_MIX_MAC(acc23, l2, r3);
        RUY_MIX_MAC(acc30, l3, r0); RUY_MIX_MAC(acc31, l3, r1);
        RUY_MIX_MAC(acc32, l3, r2); RUY_MIX_MAC(acc33, l3, r3);
      }

      // Horizontal reduction: two rounds of vpaddq_s32 collapse 4×int32
      // partial sums → [sum_row0, sum_row1, sum_row2, sum_row3] per column.
      int32x4_t p00=vpaddq_s32(acc00,acc10), p20=vpaddq_s32(acc20,acc30);
      int32x4_t p01=vpaddq_s32(acc01,acc11), p21=vpaddq_s32(acc21,acc31);
      int32x4_t p02=vpaddq_s32(acc02,acc12), p22=vpaddq_s32(acc22,acc32);
      int32x4_t p03=vpaddq_s32(acc03,acc13), p23=vpaddq_s32(acc23,acc33);
      int32x4_t s0=vpaddq_s32(p00,p20), s1=vpaddq_s32(p01,p21);
      int32x4_t s2=vpaddq_s32(p02,p22), s3=vpaddq_s32(p03,p23);

      // Post-accumulation: prod_zp_depth, bias, zero-point sums.
      const std::uint8_t flags = params.flags;
      bool ch_col = (flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) != 0;

      int32x4_t zp_depth = vdupq_n_s32(params.prod_zp_depth);
      s0 = vaddq_s32(s0, zp_depth); s1 = vaddq_s32(s1, zp_depth);
      s2 = vaddq_s32(s2, zp_depth); s3 = vaddq_s32(s3, zp_depth);

      if (flags & RUY_ASM_FLAG_HAS_BIAS) {
        if (ch_col) {
          s0 = vaddq_s32(s0, vdupq_n_s32(params.bias[col + 0]));
          s1 = vaddq_s32(s1, vdupq_n_s32(params.bias[col + 1]));
          s2 = vaddq_s32(s2, vdupq_n_s32(params.bias[col + 2]));
          s3 = vaddq_s32(s3, vdupq_n_s32(params.bias[col + 3]));
        } else {
          int32x4_t bias_vec = vld1q_s32(&params.bias[row]);
          s0=vaddq_s32(s0,bias_vec); s1=vaddq_s32(s1,bias_vec);
          s2=vaddq_s32(s2,bias_vec); s3=vaddq_s32(s3,bias_vec);
        }
      }

      if (flags & RUY_ASM_FLAG_HAS_RHS_SUMS) {
        int32x4_t lzp = vdupq_n_s32(params.lhs_zero_point);
        s0 = vmlsq_s32(s0, lzp, vdupq_n_s32(params.rhs_sums[col + 0]));
        s1 = vmlsq_s32(s1, lzp, vdupq_n_s32(params.rhs_sums[col + 1]));
        s2 = vmlsq_s32(s2, lzp, vdupq_n_s32(params.rhs_sums[col + 2]));
        s3 = vmlsq_s32(s3, lzp, vdupq_n_s32(params.rhs_sums[col + 3]));
      }

      if (flags & RUY_ASM_FLAG_HAS_LHS_SUMS) {
        int32x4_t rzp = vdupq_n_s32(params.rhs_zero_point);
        int32x4_t lhs_sums_vec = vld1q_s32(&params.lhs_sums[row]);
        int32x4_t correction = vmulq_s32(rzp, lhs_sums_vec);
        s0=vsubq_s32(s0,correction); s1=vsubq_s32(s1,correction);
        s2=vsubq_s32(s2,correction); s3=vsubq_s32(s3,correction);
      }

      // Apply multiplier, dst_zero_point, clamp, and store.
      // Mixed-precision output is always int16 — handle analogously to
      // Kernel8bitNeonImpl's int16 branch (int32 domain, no int8 narrowing).
      int fit_r = std::min(params.dst_rows - row, 4);
      int fit_c = std::min(params.dst_cols - col, 4);
      int r_off = row - params.start_row;
      int c_off = col - params.start_col;

      if (params.dst_type_id != RUY_ASM_TYPE_ID_INT32) {
        bool is_perchannel = (flags & RUY_ASM_FLAG_HAS_PERCHANNEL) != 0;
        if (is_perchannel && ch_col) {
          s0 = ApplyMultiplierVec4(s0,
               vdupq_n_s32(params.multiplier_fixedpoint[col+0]),
               vdupq_n_s32(params.multiplier_exponent[col+0]));
          s1 = ApplyMultiplierVec4(s1,
               vdupq_n_s32(params.multiplier_fixedpoint[col+1]),
               vdupq_n_s32(params.multiplier_exponent[col+1]));
          s2 = ApplyMultiplierVec4(s2,
               vdupq_n_s32(params.multiplier_fixedpoint[col+2]),
               vdupq_n_s32(params.multiplier_exponent[col+2]));
          s3 = ApplyMultiplierVec4(s3,
               vdupq_n_s32(params.multiplier_fixedpoint[col+3]),
               vdupq_n_s32(params.multiplier_exponent[col+3]));
        } else if (is_perchannel && !ch_col) {
          int32x4_t fp_vec = vld1q_s32(&params.multiplier_fixedpoint[row]);
          int32x4_t ep_vec = vld1q_s32(&params.multiplier_exponent[row]);
          s0=ApplyMultiplierVec4(s0,fp_vec,ep_vec);
          s1=ApplyMultiplierVec4(s1,fp_vec,ep_vec);
          s2=ApplyMultiplierVec4(s2,fp_vec,ep_vec);
          s3=ApplyMultiplierVec4(s3,fp_vec,ep_vec);
        } else {
          int32x4_t fp_vec = vdupq_n_s32(params.multiplier_fixedpoint[0]);
          int32x4_t ep_vec = vdupq_n_s32(params.multiplier_exponent[0]);
          s0=ApplyMultiplierVec4(s0,fp_vec,ep_vec);
          s1=ApplyMultiplierVec4(s1,fp_vec,ep_vec);
          s2=ApplyMultiplierVec4(s2,fp_vec,ep_vec);
          s3=ApplyMultiplierVec4(s3,fp_vec,ep_vec);
        }

        int32x4_t dzp32 = vdupq_n_s32(params.dst_zero_point);
        int32x4_t cmin32 = vdupq_n_s32(params.clamp_min);
        int32x4_t cmax32 = vdupq_n_s32(params.clamp_max);
        auto clamp32 = [&](int32x4_t v) -> int32x4_t {
          v = vaddq_s32(v, dzp32);
          v = vmaxq_s32(v, cmin32);
          v = vminq_s32(v, cmax32);
          return v;
        };
        s0=clamp32(s0); s1=clamp32(s1); s2=clamp32(s2); s3=clamp32(s3);

        std::uint8_t* base = static_cast<std::uint8_t*>(params.dst_base_ptr) +
                             c_off * dst_stride_bytes + r_off * 2;
        int32_t buf[4][4];
        vst1q_s32(buf[0],s0); vst1q_s32(buf[1],s1);
        vst1q_s32(buf[2],s2); vst1q_s32(buf[3],s3);
        for (int c = 0; c < fit_c; ++c)
          for (int r = 0; r < fit_r; ++r)
            reinterpret_cast<std::int16_t*>(
                base + c * dst_stride_bytes)[r] =
                static_cast<std::int16_t>(buf[c][r]);
      } else {
        std::uint8_t* base = static_cast<std::uint8_t*>(params.dst_base_ptr) +
                             c_off * dst_stride_bytes + r_off * 4;
        int32_t buf[4][4];
        vst1q_s32(buf[0],s0); vst1q_s32(buf[1],s1);
        vst1q_s32(buf[2],s2); vst1q_s32(buf[3],s3);
        for (int c = 0; c < fit_c; ++c)
          for (int r = 0; r < fit_r; ++r)
            reinterpret_cast<std::int32_t*>(
                base + c * dst_stride_bytes)[r] = buf[c][r];
      }

      rhs_col_ptr += 4 * params.rhs_stride;
    }
    lhs_col_ptr += 4 * (params.lhs_stride * lhs_elem_bytes);
  }
}

// One iteration of the depth loop: accumulate one 16-deep slice of a single
// (row, col) cell into a 4×int32 partial-sum vector using the three-instruction
// GAS pattern: smull (low 8 int8→int16), smlal2 (high 8), sadalp (→int32).
// The *16* depth values of lhs_row and rhs_col are multiplied pairwise and
// horizontally summed; the result accumulates into `acc`.
#define RUY_NEON8BIT_MAC(acc_, lhs_row_, rhs_col_)                        \
    do {                                                                   \
        int16x8_t prod_ = vmull_s8(vget_low_s8(lhs_row_),                 \
                                   vget_low_s8(rhs_col_));                 \
        prod_ = vmlal_high_s8(prod_, lhs_row_, rhs_col_);                 \
        acc_ = vpadalq_s16(acc_, prod_);                                   \
    } while (0)

static void Kernel8bitNeonImpl(const KernelParams8bit<4, 4>& params) {
  const int depth = params.depth;
  const int dst_stride_bytes = params.dst_stride;

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_base =
      static_cast<const std::int8_t*>(params.rhs_base_ptr);

  for (int row = params.start_row; row <= params.last_row; row += 4) {
    const std::int8_t* rhs_col_ptr = rhs_col_base;
    for (int col = params.start_col; col <= params.last_col; col += 4) {
      // 16 int32 accumulators: acc[r][c] holds partial sums for row r, col c.
      // Each is a 4-element int32 vector; the four elements are horizontal
      // partials that get reduced after the depth loop.
      int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0);
      int32x4_t acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
      int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0);
      int32x4_t acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
      int32x4_t acc20 = vdupq_n_s32(0), acc21 = vdupq_n_s32(0);
      int32x4_t acc22 = vdupq_n_s32(0), acc23 = vdupq_n_s32(0);
      int32x4_t acc30 = vdupq_n_s32(0), acc31 = vdupq_n_s32(0);
      int32x4_t acc32 = vdupq_n_s32(0), acc33 = vdupq_n_s32(0);

      const std::int8_t* lhs_ptr = lhs_col_ptr;
      const std::int8_t* rhs_ptr = rhs_col_ptr;

      // Inner loop: 16 depth values per iteration.
      // Layout: 4 rows × 16 bytes for LHS, 4 cols × 16 bytes for RHS.
      for (int d = 0; d < depth; d += 16) {
        int8x16_t l0 = vld1q_s8(lhs_ptr);       // row 0, depths [d..d+15]
        int8x16_t l1 = vld1q_s8(lhs_ptr + 16);  // row 1
        int8x16_t l2 = vld1q_s8(lhs_ptr + 32);  // row 2
        int8x16_t l3 = vld1q_s8(lhs_ptr + 48);  // row 3
        int8x16_t r0 = vld1q_s8(rhs_ptr);        // col 0
        int8x16_t r1 = vld1q_s8(rhs_ptr + 16);   // col 1
        int8x16_t r2 = vld1q_s8(rhs_ptr + 32);   // col 2
        int8x16_t r3 = vld1q_s8(rhs_ptr + 48);   // col 3

        RUY_NEON8BIT_MAC(acc00, l0, r0); RUY_NEON8BIT_MAC(acc01, l0, r1);
        RUY_NEON8BIT_MAC(acc02, l0, r2); RUY_NEON8BIT_MAC(acc03, l0, r3);
        RUY_NEON8BIT_MAC(acc10, l1, r0); RUY_NEON8BIT_MAC(acc11, l1, r1);
        RUY_NEON8BIT_MAC(acc12, l1, r2); RUY_NEON8BIT_MAC(acc13, l1, r3);
        RUY_NEON8BIT_MAC(acc20, l2, r0); RUY_NEON8BIT_MAC(acc21, l2, r1);
        RUY_NEON8BIT_MAC(acc22, l2, r2); RUY_NEON8BIT_MAC(acc23, l2, r3);
        RUY_NEON8BIT_MAC(acc30, l3, r0); RUY_NEON8BIT_MAC(acc31, l3, r1);
        RUY_NEON8BIT_MAC(acc32, l3, r2); RUY_NEON8BIT_MAC(acc33, l3, r3);

        lhs_ptr += 64;
        rhs_ptr += 64;
      }

      // Horizontal reduction: collapse 4-element partial-sum vectors to scalars.
      // Two rounds of addp give a single 4×int32 vector per column holding
      // [sum_row0, sum_row1, sum_row2, sum_row3].
      int32x4_t s0, s1, s2, s3;
      // Round 1: pairwise-add adjacent rows within each column group
      int32x4_t p00 = vpaddq_s32(acc00, acc10);
      int32x4_t p20 = vpaddq_s32(acc20, acc30);
      int32x4_t p01 = vpaddq_s32(acc01, acc11);
      int32x4_t p21 = vpaddq_s32(acc21, acc31);
      int32x4_t p02 = vpaddq_s32(acc02, acc12);
      int32x4_t p22 = vpaddq_s32(acc22, acc32);
      int32x4_t p03 = vpaddq_s32(acc03, acc13);
      int32x4_t p23 = vpaddq_s32(acc23, acc33);
      // Round 2: final horizontal reduction — each vector now holds
      // [sum_row0, sum_row1, sum_row2, sum_row3] for one output column.
      s0 = vpaddq_s32(p00, p20);  // col 0: [r0c0, r1c0, r2c0, r3c0]
      s1 = vpaddq_s32(p01, p21);  // col 1
      s2 = vpaddq_s32(p02, p22);  // col 2
      s3 = vpaddq_s32(p03, p23);  // col 3

      // Post-accumulation: prod_zp_depth, bias, zero-point sums.
      const std::uint8_t flags = params.flags;
      bool ch_col = (flags & RUY_ASM_FLAG_CHANNEL_DIMENSION_IS_COL) != 0;

      int32x4_t zp_depth = vdupq_n_s32(params.prod_zp_depth);
      s0 = vaddq_s32(s0, zp_depth);
      s1 = vaddq_s32(s1, zp_depth);
      s2 = vaddq_s32(s2, zp_depth);
      s3 = vaddq_s32(s3, zp_depth);

      if (flags & RUY_ASM_FLAG_HAS_BIAS) {
        if (ch_col) {
          // Bias is per-column: each of s0..s3 gets a scalar bias value.
          s0 = vaddq_s32(s0, vdupq_n_s32(params.bias[col + 0]));
          s1 = vaddq_s32(s1, vdupq_n_s32(params.bias[col + 1]));
          s2 = vaddq_s32(s2, vdupq_n_s32(params.bias[col + 2]));
          s3 = vaddq_s32(s3, vdupq_n_s32(params.bias[col + 3]));
        } else {
          // Bias is per-row: all columns share the same 4-row bias vector.
          int32x4_t bias_vec = vld1q_s32(&params.bias[row]);
          s0 = vaddq_s32(s0, bias_vec);
          s1 = vaddq_s32(s1, bias_vec);
          s2 = vaddq_s32(s2, bias_vec);
          s3 = vaddq_s32(s3, bias_vec);
        }
      }

      if (flags & RUY_ASM_FLAG_HAS_RHS_SUMS) {
        // Subtract lhs_zero_point * rhs_sums[col]
        int32x4_t lzp = vdupq_n_s32(params.lhs_zero_point);
        s0 = vmlsq_s32(s0, lzp, vdupq_n_s32(params.rhs_sums[col + 0]));
        s1 = vmlsq_s32(s1, lzp, vdupq_n_s32(params.rhs_sums[col + 1]));
        s2 = vmlsq_s32(s2, lzp, vdupq_n_s32(params.rhs_sums[col + 2]));
        s3 = vmlsq_s32(s3, lzp, vdupq_n_s32(params.rhs_sums[col + 3]));
      }

      if (flags & RUY_ASM_FLAG_HAS_LHS_SUMS) {
        // Subtract rhs_zero_point * lhs_sums[row..row+3]
        int32x4_t rzp = vdupq_n_s32(params.rhs_zero_point);
        int32x4_t lhs_sums_vec = vld1q_s32(&params.lhs_sums[row]);
        int32x4_t correction = vmulq_s32(rzp, lhs_sums_vec);
        s0 = vsubq_s32(s0, correction);
        s1 = vsubq_s32(s1, correction);
        s2 = vsubq_s32(s2, correction);
        s3 = vsubq_s32(s3, correction);
      }

      // For non-int32 output: apply multiplier, add dst_zero_point, clamp.
      if (params.dst_type_id != RUY_ASM_TYPE_ID_INT32) {
        bool is_perchannel = (flags & RUY_ASM_FLAG_HAS_PERCHANNEL) != 0;

        auto get_fp = [&](int idx) -> int32x4_t {
          return vdupq_n_s32(params.multiplier_fixedpoint[idx]);
        };
        auto get_ep = [&](int idx) -> int32x4_t {
          return vdupq_n_s32(params.multiplier_exponent[idx]);
        };

        if (is_perchannel) {
          if (ch_col) {
            // Per-column: each column has a distinct scalar multiplier.
            s0 = ApplyMultiplierVec4(s0, get_fp(col+0), get_ep(col+0));
            s1 = ApplyMultiplierVec4(s1, get_fp(col+1), get_ep(col+1));
            s2 = ApplyMultiplierVec4(s2, get_fp(col+2), get_ep(col+2));
            s3 = ApplyMultiplierVec4(s3, get_fp(col+3), get_ep(col+3));
          } else {
            // Per-row: each row has a distinct scalar multiplier; all columns
            // use the same 4-element vector of per-row multipliers.
            int32x4_t fp_vec = vld1q_s32(&params.multiplier_fixedpoint[row]);
            int32x4_t ep_vec = vld1q_s32(&params.multiplier_exponent[row]);
            s0 = ApplyMultiplierVec4(s0, fp_vec, ep_vec);
            s1 = ApplyMultiplierVec4(s1, fp_vec, ep_vec);
            s2 = ApplyMultiplierVec4(s2, fp_vec, ep_vec);
            s3 = ApplyMultiplierVec4(s3, fp_vec, ep_vec);
          }
        } else {
          int32x4_t fp_vec = vdupq_n_s32(params.multiplier_fixedpoint[0]);
          int32x4_t ep_vec = vdupq_n_s32(params.multiplier_exponent[0]);
          s0 = ApplyMultiplierVec4(s0, fp_vec, ep_vec);
          s1 = ApplyMultiplierVec4(s1, fp_vec, ep_vec);
          s2 = ApplyMultiplierVec4(s2, fp_vec, ep_vec);
          s3 = ApplyMultiplierVec4(s3, fp_vec, ep_vec);
        }

        // Compute fit dims and destination base pointer (shared by all output types).
        int fit_r = std::min(params.dst_rows - row, 4);
        int fit_c = std::min(params.dst_cols - col, 4);
        int r_off = row - params.start_row;
        int c_off = col - params.start_col;

        if (params.dst_type_id == RUY_ASM_TYPE_ID_INT16) {
          // int16 output: add dst_zero_point and clamp in int32 domain, then
          // narrow to int16 and store.  Keep the same scalar-equivalent path so
          // results are bit-exact with the reference (no extra int8-saturation).
          int32x4_t dzp32 = vdupq_n_s32(params.dst_zero_point);
          int32x4_t cmin32 = vdupq_n_s32(params.clamp_min);
          int32x4_t cmax32 = vdupq_n_s32(params.clamp_max);
          auto clamp32 = [&](int32x4_t v) -> int32x4_t {
            v = vaddq_s32(v, dzp32);
            v = vmaxq_s32(v, cmin32);
            v = vminq_s32(v, cmax32);
            return v;
          };
          s0 = clamp32(s0); s1 = clamp32(s1);
          s2 = clamp32(s2); s3 = clamp32(s3);

          std::uint8_t* base = static_cast<std::uint8_t*>(params.dst_base_ptr) +
                               c_off * dst_stride_bytes + r_off * 2;
          int32_t buf[4][4];
          vst1q_s32(buf[0], s0); vst1q_s32(buf[1], s1);
          vst1q_s32(buf[2], s2); vst1q_s32(buf[3], s3);
          for (int c = 0; c < fit_c; ++c)
            for (int r = 0; r < fit_r; ++r)
              reinterpret_cast<std::int16_t*>(
                  base + c * dst_stride_bytes)[r] =
                  static_cast<std::int16_t>(buf[c][r]);
        } else {
          // int8 / uint8 output: narrow int32 → int16 → int8 with saturating
          // arithmetic, adding dst_zero_point in int16 domain to match GAS kernel.
          int16x4_t n0 = vqmovn_s32(s0);  // 4×int32 → 4×int16 (saturating)
          int16x4_t n1 = vqmovn_s32(s1);
          int16x4_t n2 = vqmovn_s32(s2);
          int16x4_t n3 = vqmovn_s32(s3);
          int16x8_t w01 = vcombine_s16(n0, n1);
          int16x8_t w23 = vcombine_s16(n2, n3);
          int16x8_t dzp16 = vdupq_n_s16(
              static_cast<std::int16_t>(params.dst_zero_point));
          w01 = vqaddq_s16(w01, dzp16);
          w23 = vqaddq_s16(w23, dzp16);
          std::int16_t cmin16 = static_cast<std::int16_t>(params.clamp_min);
          std::int16_t cmax16 = static_cast<std::int16_t>(params.clamp_max);
          w01 = vmaxq_s16(w01, vdupq_n_s16(cmin16));
          w01 = vminq_s16(w01, vdupq_n_s16(cmax16));
          w23 = vmaxq_s16(w23, vdupq_n_s16(cmin16));
          w23 = vminq_s16(w23, vdupq_n_s16(cmax16));

          std::uint8_t* base = static_cast<std::uint8_t*>(params.dst_base_ptr) +
                               c_off * dst_stride_bytes + r_off;
          if (params.dst_type_id == RUY_ASM_TYPE_ID_UINT8) {
            uint8x8_t u01 = vqmovun_s16(w01);
            uint8x8_t u23 = vqmovun_s16(w23);
            std::uint8_t buf[16];
            vst1_u8(buf,     u01);
            vst1_u8(buf + 8, u23);
            for (int c = 0; c < fit_c; ++c)
              for (int r = 0; r < fit_r; ++r)
                (base + c * dst_stride_bytes)[r] = buf[c * 4 + r];
          } else {
            int8x8_t b01 = vqmovn_s16(w01);
            int8x8_t b23 = vqmovn_s16(w23);
            std::int8_t buf[16];
            vst1_s8(buf,     b01);
            vst1_s8(buf + 8, b23);
            for (int c = 0; c < fit_c; ++c)
              for (int r = 0; r < fit_r; ++r)
                reinterpret_cast<std::int8_t*>(
                    base + c * dst_stride_bytes)[r] = buf[c * 4 + r];
          }
        }
      } else {
        // int32 output: store raw accumulators, no multiplier.
        int fit_r = std::min(params.dst_rows - row, 4);
        int fit_c = std::min(params.dst_cols - col, 4);
        int r_off = row - params.start_row;
        int c_off = col - params.start_col;
        std::uint8_t* base = static_cast<std::uint8_t*>(params.dst_base_ptr) +
                             c_off * dst_stride_bytes + r_off * 4;
        int32_t cols_buf[4][4];
        vst1q_s32(cols_buf[0], s0);
        vst1q_s32(cols_buf[1], s1);
        vst1q_s32(cols_buf[2], s2);
        vst1q_s32(cols_buf[3], s3);
        for (int c = 0; c < fit_c; ++c)
          for (int r = 0; r < fit_r; ++r)
            reinterpret_cast<std::int32_t*>(
                base + c * dst_stride_bytes)[r] = cols_buf[c][r];
      }

      rhs_col_ptr += 4 * params.rhs_stride;
    }
    lhs_col_ptr += 4 * params.lhs_stride;
  }
}

void Kernel8bitNeon(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon, MSVC 4x4)");
  Kernel8bitNeonImpl(params);
}
void Kernel8bitNeon1Col(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon1Col, MSVC 4x4)");
  Kernel8bitNeonImpl(params);
}
void Kernel8bitNeonA55ish(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeonA55ish, MSVC 4x4)");
  Kernel8bitNeonImpl(params);
}

// Mixed-precision kernels: i8×i16→i16 and i16×i8→i16.
// Both use Kernel8bitNeonMixedImpl<> with tile 4×4 and depth step 8.
void Kernel8bitNeonMixedInt16Lhs(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon mixed i16xint8, MSVC 4x4)");
  Kernel8bitNeonMixedImpl<false>(params);  // LHS=int16, RHS=int8
}
void Kernel8bitNeonMixedInt16Rhs(const KernelParams8bit<4, 4>& params) {
  profiler::ScopeLabel label("Kernel (kNeon mixed int8xi16, MSVC 4x4)");
  Kernel8bitNeonMixedImpl<true>(params);   // LHS=int8,  RHS=int16
}

// Pack8bitColMajorForNeon / ForNeonDotprod:
// Loads 16 bytes from each of 4 source ptrs, XORs with input_xor, stores
// interleaved to packed_ptr, accumulates row sums.
void Pack8bitColMajorForNeon(const void* src_ptr0, const void* src_ptr1,
                              const void* src_ptr2, const void* src_ptr3,
                              int src_inc0, int src_inc1, int src_inc2,
                              int src_inc3, int src_rows, int src_zero_point,
                              std::int8_t* packed_ptr, std::int32_t* sums_ptr,
                              int input_xor) {
  profiler::ScopeLabel label("Pack (kNeon, MSVC NEON)");
  const std::uint8_t* s0 = static_cast<const std::uint8_t*>(src_ptr0);
  const std::uint8_t* s1 = static_cast<const std::uint8_t*>(src_ptr1);
  const std::uint8_t* s2 = static_cast<const std::uint8_t*>(src_ptr2);
  const std::uint8_t* s3 = static_cast<const std::uint8_t*>(src_ptr3);
  uint8x16_t xorv = vdupq_n_u8(static_cast<std::uint8_t>(input_xor));

  int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0);
  int32x4_t sum2 = vdupq_n_s32(0), sum3 = vdupq_n_s32(0);

  int r = 0;
  for (; r + 16 <= src_rows; r += 16) {
    uint8x16_t v0 = veorq_u8(vld1q_u8(s0), xorv);
    uint8x16_t v1 = veorq_u8(vld1q_u8(s1), xorv);
    uint8x16_t v2 = veorq_u8(vld1q_u8(s2), xorv);
    uint8x16_t v3 = veorq_u8(vld1q_u8(s3), xorv);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr),      v0);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr) + 16, v1);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr) + 32, v2);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr) + 48, v3);
    packed_ptr += 64;
    sum0 = vpadalq_s16(sum0, vpaddlq_s8(vreinterpretq_s8_u8(v0)));
    sum1 = vpadalq_s16(sum1, vpaddlq_s8(vreinterpretq_s8_u8(v1)));
    sum2 = vpadalq_s16(sum2, vpaddlq_s8(vreinterpretq_s8_u8(v2)));
    sum3 = vpadalq_s16(sum3, vpaddlq_s8(vreinterpretq_s8_u8(v3)));
    s0 += src_inc0;
    s1 += src_inc1;
    s2 += src_inc2;
    s3 += src_inc3;
  }
  // Tail: depth rows remaining (< 16 per group, up to 15 valid depth elements).
  // Must store a FULL 64-byte (4×16) block with zero-padding to match the
  // GAS kernel's layout:
  //   col i at packed_ptr + i*16 (16 bytes), depth element k at byte k (0-based).
  // The GAS kernel zero-pads v0..v3 with src_zero_point, fills valid lanes,
  // XORs, then str q4/q5/q6/q7 at {0,16,32,48}.
  if (r < src_rows) {
    int rem = src_rows - r;
    // Build 4 zero-padded vectors (16 bytes each = 16 depth slots).
    std::uint8_t zp = static_cast<std::uint8_t>(src_zero_point);
    uint8x16_t zp16 = vdupq_n_u8(zp);
    uint8x16_t w0 = zp16, w1 = zp16, w2 = zp16, w3 = zp16;
    // Fill valid depth lanes from source.
    std::uint8_t buf0[16], buf1[16], buf2[16], buf3[16];
    vst1q_u8(buf0, zp16); vst1q_u8(buf1, zp16);
    vst1q_u8(buf2, zp16); vst1q_u8(buf3, zp16);
    const std::uint8_t* sp[4] = {s0, s1, s2, s3};
    for (int k = 0; k < rem; ++k) {
      buf0[k] = *sp[0]++;
      buf1[k] = *sp[1]++;
      buf2[k] = *sp[2]++;
      buf3[k] = *sp[3]++;
    }
    // XOR and store.
    w0 = veorq_u8(vld1q_u8(buf0), xorv);
    w1 = veorq_u8(vld1q_u8(buf1), xorv);
    w2 = veorq_u8(vld1q_u8(buf2), xorv);
    w3 = veorq_u8(vld1q_u8(buf3), xorv);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr),      w0);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr) + 16, w1);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr) + 32, w2);
    vst1q_u8(reinterpret_cast<std::uint8_t*>(packed_ptr) + 48, w3);
    packed_ptr += 64;
    if (sums_ptr) {
      sum0 = vpadalq_s16(sum0, vpaddlq_s8(vreinterpretq_s8_u8(w0)));
      sum1 = vpadalq_s16(sum1, vpaddlq_s8(vreinterpretq_s8_u8(w1)));
      sum2 = vpadalq_s16(sum2, vpaddlq_s8(vreinterpretq_s8_u8(w2)));
      sum3 = vpadalq_s16(sum3, vpaddlq_s8(vreinterpretq_s8_u8(w3)));
    }
  }
  if (sums_ptr) {
    // Use = (store), not += (add), because the sums buffer may contain
    // uninitialized / stale values from a previous Mul call.  The allocator
    // is a bump-allocator that resets its pointer without zeroing memory, so
    // the sums buffer can hold garbage when we first write it.  The GAS and
    // generic scalar pack functions both store (not add), so we must too.
    sums_ptr[0] = vaddvq_s32(sum0);
    sums_ptr[1] = vaddvq_s32(sum1);
    sums_ptr[2] = vaddvq_s32(sum2);
    sums_ptr[3] = vaddvq_s32(sum3);
  }
}

void Pack8bitColMajorForNeonA55ish(
    const void* src_ptr0, const void* src_ptr1, const void* src_ptr2,
    const void* src_ptr3, int src_inc0, int src_inc1, int src_inc2,
    int src_inc3, int src_rows, int src_zero_point, std::int8_t* packed_ptr,
    std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label("Pack (kNeonA55ish, MSVC NEON)");
  Pack8bitColMajorForNeon(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0,
                           src_inc1, src_inc2, src_inc3, src_rows, src_zero_point,
                           packed_ptr, sums_ptr, input_xor);
}

void Pack8bitColMajorForNeonDotprod(
    const void* src_ptr0, const void* src_ptr1, const void* src_ptr2,
    const void* src_ptr3, int src_inc0, int src_inc1, int src_inc2,
    int src_inc3, int src_rows, int src_zero_point, std::int8_t* packed_ptr,
    std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label("Pack (kNeonDotprod, MSVC NEON)");
  // For each 4-depth group, store 16 bytes = {col0[d..d+3], col1[d..d+3], col2[d..d+3], col3[d..d+3]}.
  // Output stride between consecutive 4-depth groups: 32 bytes (the companion
  // group of 4 cols sits at offset+16 from the same base, leaving a 32-byte period).
  //
  // The GAS kernel implements this via 16-row transpose blocks:
  //   For 16 depth rows loaded in v0..v3 (one 16-byte vector per col):
  //   trn1/trn2 produces 4 transposed 4-depth groups, stored at {0, 32, 64, 96}, then += 128.

  const std::uint8_t* s0 = static_cast<const std::uint8_t*>(src_ptr0);
  const std::uint8_t* s1 = static_cast<const std::uint8_t*>(src_ptr1);
  const std::uint8_t* s2 = static_cast<const std::uint8_t*>(src_ptr2);
  const std::uint8_t* s3 = static_cast<const std::uint8_t*>(src_ptr3);
  uint8x16_t xorv = vdupq_n_u8(static_cast<std::uint8_t>(input_xor));
  // Per-row sums: sum[j] = sum of all packed bytes for GEMM row j (j=0..3).
  // Matches GAS: sdot sum.4s, g_k.16b, ones.16b
  //   where g_k[4j..4j+3] = row_j depths k*4..k*4+3.
  // ones = {1,1,...,1}, so vdotq_s32(sum, g_k, ones)[j] += sum(g_k[4j..4j+3]).
  int32x4_t sum_rows = vdupq_n_s32(0);  // 4 per-row sums (rows 0..3 of this 4-col block)
  int8x16_t ones = vdupq_n_s8(1);

  int r = 0;
  for (; r + 16 <= src_rows; r += 16) {
    uint8x16_t v0 = vld1q_u8(s0);
    uint8x16_t v1 = vld1q_u8(s1);
    uint8x16_t v2 = vld1q_u8(s2);
    uint8x16_t v3 = vld1q_u8(s3);
    RUY_DOTPROD_PACK_TRANSPOSE_STORE(v0, v1, v2, v3, packed_ptr,
                                     (sums_ptr != nullptr));
    s0 += src_inc0; s1 += src_inc1; s2 += src_inc2; s3 += src_inc3;
  }
  // Tail: remaining depth rows (< 16). Zero-pad with src_zero_point.
  // The GAS kernel applies SDOT only for groups 0..ceil(rem/4)-1 (matches cmp/ble
  // chain). We do the same: inline the tail without the macro so we can gate
  // individual SDOT calls. The extra stored groups (kernel never reads them) are
  // harmless but must NOT contribute to sums.
  if (r < src_rows) {
    int rem = src_rows - r;
    int num_valid_groups = (rem + 3) / 4;  // 1..4
    std::uint8_t zp = static_cast<std::uint8_t>(src_zero_point);
    uint8x16_t zp16 = vdupq_n_u8(zp);
    std::uint8_t buf0[16], buf1[16], buf2[16], buf3[16];
    vst1q_u8(buf0, zp16); vst1q_u8(buf1, zp16);
    vst1q_u8(buf2, zp16); vst1q_u8(buf3, zp16);
    for (int k = 0; k < rem; ++k) {
      buf0[k] = s0[k]; buf1[k] = s1[k];
      buf2[k] = s2[k]; buf3[k] = s3[k];
    }
    // XOR and transpose — same logic as the macro body.
    uint8x16_t e0t = veorq_u8(vld1q_u8(buf0), xorv);
    uint8x16_t e1t = veorq_u8(vld1q_u8(buf1), xorv);
    uint8x16_t e2t = veorq_u8(vld1q_u8(buf2), xorv);
    uint8x16_t e3t = veorq_u8(vld1q_u8(buf3), xorv);
    int32x4x2_t t01t = vtrnq_s32(vreinterpretq_s32_u8(e0t),
                                   vreinterpretq_s32_u8(e1t));
    int32x4x2_t t23t = vtrnq_s32(vreinterpretq_s32_u8(e2t),
                                   vreinterpretq_s32_u8(e3t));
    int8x16_t tg0 = vreinterpretq_s8_s64(vcombine_s64(
      vget_low_s64(vreinterpretq_s64_s32(t01t.val[0])),
      vget_low_s64(vreinterpretq_s64_s32(t23t.val[0]))));
    int8x16_t tg1 = vreinterpretq_s8_s64(vcombine_s64(
      vget_low_s64(vreinterpretq_s64_s32(t01t.val[1])),
      vget_low_s64(vreinterpretq_s64_s32(t23t.val[1]))));
    int8x16_t tg2 = vreinterpretq_s8_s64(vcombine_s64(
      vget_high_s64(vreinterpretq_s64_s32(t01t.val[0])),
      vget_high_s64(vreinterpretq_s64_s32(t23t.val[0]))));
    int8x16_t tg3 = vreinterpretq_s8_s64(vcombine_s64(
      vget_high_s64(vreinterpretq_s64_s32(t01t.val[1])),
      vget_high_s64(vreinterpretq_s64_s32(t23t.val[1]))));
    // Accumulate per-row sums only for groups that the kernel actually reads.
    // Matches GAS cmp/ble chain: stop after group ceil(rem/4)-1.
    if (sums_ptr) {
      /* group 0 always valid */
                               sum_rows = vdotq_s32(sum_rows, tg0, ones);
      if (num_valid_groups >= 2) sum_rows = vdotq_s32(sum_rows, tg1, ones);
      if (num_valid_groups >= 3) sum_rows = vdotq_s32(sum_rows, tg2, ones);
      if (num_valid_groups >= 4) sum_rows = vdotq_s32(sum_rows, tg3, ones);
    }
    // Store only valid groups, matching GAS cmp/ble chain.
    // For rem <= 4: only g0 written (16 bytes). No pointer advance.
    // For rem <= 8: g0+g1. For rem <= 12: g0+g1+g2. For rem > 12: all 4 + advance.
                                   vst1q_s8(packed_ptr,      tg0);
    if (num_valid_groups >= 2)     vst1q_s8(packed_ptr + 32, tg1);
    if (num_valid_groups >= 3)     vst1q_s8(packed_ptr + 64, tg2);
    if (num_valid_groups >= 4) {
                                   vst1q_s8(packed_ptr + 96, tg3);
                                   packed_ptr += 128;
    }
  }

  if (sums_ptr) {
    // Store 4 per-row sums. Use vst1q (= store, not add) so stale allocator
    // values from a prior Mul call are overwritten.
    // Matches GAS: after sdot-with-ones on transposed groups,
    //   stores {row0_sum, row1_sum, row2_sum, row3_sum}.
    vst1q_s32(sums_ptr, sum_rows);
  }
}

void Pack8bitColMajorForNeonDotprodA55ish(
    const void* src_ptr0, const void* src_ptr1, const void* src_ptr2,
    const void* src_ptr3, int src_inc0, int src_inc1, int src_inc2,
    int src_inc3, int src_rows, int src_zero_point, std::int8_t* packed_ptr,
    std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label("Pack (kNeonDotprodA55ish, MSVC NEON)");
  Pack8bitColMajorForNeonDotprod(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                  src_inc0, src_inc1, src_inc2, src_inc3,
                                  src_rows, src_zero_point,
                                  packed_ptr, sums_ptr, input_xor);
}

// Pack8bitRowMajorForNeonDotprod has an extra packed_stride parameter.
// Output layout (matches GAS zip1/zip1/zip1.8h/zip2.8h + str q2/q3):
//   For each group of 8 src columns at offset c:
//     packed[0+4*j + i] = row_i[c+j]  for i in 0..3, j in 0..7
//   i.e. column-major within each 4-row group of cols.
//   Then packed_ptr advances by packed_stride * 8 bytes (to the next
//   group of 8 depth columns).
void Pack8bitRowMajorForNeonDotprod(
    const void* src_ptr0, const void* src_ptr1, const void* src_ptr2,
    const void* src_ptr3, int src_inc0, int src_inc1, int src_inc2,
    int src_inc3, int src_cols, int src_zero_point, std::int8_t* packed_ptr,
    int packed_stride, std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label("Pack (kNeonDotprod row-major, MSVC NEON)");
  const std::uint8_t* sp[4] = {
      static_cast<const std::uint8_t*>(src_ptr0),
      static_cast<const std::uint8_t*>(src_ptr1),
      static_cast<const std::uint8_t*>(src_ptr2),
      static_cast<const std::uint8_t*>(src_ptr3)};
  const int inc[4] = {src_inc0, src_inc1, src_inc2, src_inc3};
  const std::int8_t xorv = static_cast<std::int8_t>(input_xor);
  const std::int8_t zp_xored = static_cast<std::int8_t>(src_zero_point ^ input_xor);
  for (int c = 0; c < src_cols; c += 8) {
    std::int8_t* col_ptr = packed_ptr;
    for (int j = 0; j < 8; ++j) {
      int ci = c + j;
      for (int i = 0; i < 4; ++i) {
        std::int8_t b;
        if (ci < src_cols)
          b = static_cast<std::int8_t>(sp[i][j] ^ xorv);
        else
          b = zp_xored;
        col_ptr[j * 4 + i] = b;
        if (sums_ptr && ci < src_cols) sums_ptr[ci] += b;
      }
    }
    packed_ptr += packed_stride * 8;
    for (int i = 0; i < 4; ++i) sp[i] += inc[i];
  }
}

// PackFloatColMajorForNeon:
// Packs float columns. Each call handles 4 src columns (4 src ptrs).
// src_inc values are in bytes (16 or 0).
//
// Output layout (must match what KernelFloatNeonImpl and the GAS kernel expect):
//   The kernel reads data as: at depth d, 8 consecutive floats = all 8 GEMM-rows.
//   Two calls to this function cover cols 0..3 (at packed_base+0) and cols 4..7
//   (at packed_base+4). Combined, packed[d*8 + c] = LHS[row_c, depth_d].
//
//   Within one 4-column call: output stride between depth steps = 8 floats (32 bytes),
//   because the companion 4-column group occupies the interleaved float positions.
//   We must TRANSPOSE the 4x4 block:
//     Input per 4 depths: v0={col0:d0..d3}, v1={col1:d0..d3}, ...
//     Output per depth:   row_d = {col0_d, col1_d, col2_d, col3_d} at packed_ptr + d*32 bytes
//   The GAS kernel achieves this with trn1/trn2 + str-with-stride-32.
void PackFloatColMajorForNeon(const float* src_ptr0, const float* src_ptr1,
                               const float* src_ptr2, const float* src_ptr3,
                               int src_inc0, int src_inc1, int src_inc2,
                               int src_inc3, int src_rows, float* packed_ptr) {
  profiler::ScopeLabel label("Pack (Float kNeon, MSVC NEON)");
  const float* s[4] = {src_ptr0, src_ptr1, src_ptr2, src_ptr3};
  // src_inc values are in bytes; convert to float strides.
  const int inc[4] = {src_inc0 / static_cast<int>(sizeof(float)),
                      src_inc1 / static_cast<int>(sizeof(float)),
                      src_inc2 / static_cast<int>(sizeof(float)),
                      src_inc3 / static_cast<int>(sizeof(float))};

  int r = 0;
  for (; r + 4 <= src_rows; r += 4) {
    float32x4_t v0 = vld1q_f32(s[0]);
    float32x4_t v1 = vld1q_f32(s[1]);
    float32x4_t v2 = vld1q_f32(s[2]);
    float32x4_t v3 = vld1q_f32(s[3]);
    RUY_PACK_FLOAT_TRANSPOSE_STORE(v0, v1, v2, v3, packed_ptr);
    for (int i = 0; i < 4; ++i) s[i] += inc[i];
  }
  // Tail: < 4 remaining depth rows.  Zero-pad to 4, transpose, store only valid rows.
  if (r < src_rows) {
    int rem = src_rows - r;
    float buf0[4] = {0.f,0.f,0.f,0.f}, buf1[4] = {0.f,0.f,0.f,0.f};
    float buf2[4] = {0.f,0.f,0.f,0.f}, buf3[4] = {0.f,0.f,0.f,0.f};
    for (int k = 0; k < rem; ++k) {
      buf0[k] = s[0][k];
      buf1[k] = s[1][k];
      buf2[k] = s[2][k];
      buf3[k] = s[3][k];
    }
    float32x4_t v0 = vld1q_f32(buf0);
    float32x4_t v1 = vld1q_f32(buf1);
    float32x4_t v2 = vld1q_f32(buf2);
    float32x4_t v3 = vld1q_f32(buf3);
    float32x4x2_t trn01 = vtrnq_f32(v0, v1);
    float32x4x2_t trn23 = vtrnq_f32(v2, v3);
    float32x4_t rows[4];
    rows[0] = vcombine_f32(vget_low_f32(trn01.val[0]), vget_low_f32(trn23.val[0]));
    rows[1] = vcombine_f32(vget_low_f32(trn01.val[1]), vget_low_f32(trn23.val[1]));
    rows[2] = vcombine_f32(vget_high_f32(trn01.val[0]), vget_high_f32(trn23.val[0]));
    rows[3] = vcombine_f32(vget_high_f32(trn01.val[1]), vget_high_f32(trn23.val[1]));
    for (int k = 0; k < rem; ++k) {
      vst1q_f32(packed_ptr, rows[k]);
      packed_ptr += 8;
    }
  }

}

void PackFloatColMajorForNeonA55ish(const float* src_ptr0,
                                     const float* src_ptr1,
                                     const float* src_ptr2,
                                     const float* src_ptr3, int src_inc0,
                                     int src_inc1, int src_inc2, int src_inc3,
                                     int src_rows, float* packed_ptr) {
  profiler::ScopeLabel label("Pack (Float kNeonA55ish, MSVC NEON)");
  PackFloatColMajorForNeon(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0,
                            src_inc1, src_inc2, src_inc3, src_rows, packed_ptr);
}

#undef RUY_NEON8BIT_MAC
#undef RUY_MIX_MAC
#undef RUY_PACK_FLOAT_TRANSPOSE_STORE
#undef RUY_DOTPROD_PACK_TRANSPOSE_STORE

}  // namespace ruy

#endif  // defined(_MSC_VER) && defined(_M_ARM64)
