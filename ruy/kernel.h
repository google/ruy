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

#ifndef RUY_RUY_KERNEL_H_
#define RUY_RUY_KERNEL_H_

#include "ruy/kernel_common.h"
#include "ruy/platform.h"

// IWYU pragma: begin_exports
#if RUY_PLATFORM_NEON
#include "ruy/kernel_arm.h"
#elif RUY_PLATFORM_X86
#include "ruy/kernel_x86.h"
#endif
// IWYU pragma: end_exports

namespace ruy {

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void RunKernelTyped(Tuning tuning, const PMat<LhsScalar>& lhs,
                    const PMat<RhsScalar>& rhs, const MulParamsType& mul_params,
                    int start_row, int start_col, int end_row, int end_col,
                    Mat<DstScalar>* dst) {
  using Kernel =
      Kernel<ThePath, LhsScalar, RhsScalar, DstScalar, MulParamsType>;
  Kernel kernel(tuning);
  using LhsLayout = typename Kernel::LhsLayout;
  using RhsLayout = typename Kernel::RhsLayout;
  // end_row and end_col may be larger than dst dimensions.
  // that is because kernels write directly to the destination matrix, whose
  // dimensions may not be a multiple of the kernel dimensions, and we try to
  // keep this annoyance localized as an implementation detail in kernels,
  // by allowing to pass rounded-up values down as far as possible.
  // These assertions encode the contract.
  RUY_DCHECK_LE(0, start_row);
  RUY_DCHECK_LE(start_row, end_row);
  RUY_DCHECK_LT(end_row, dst->layout.rows + LhsLayout::kCols);
  RUY_DCHECK_EQ((end_row - start_row) % LhsLayout::kCols, 0);
  RUY_DCHECK_LE(0, start_col);
  RUY_DCHECK_LE(start_col, end_col);
  RUY_DCHECK_LT(end_col, dst->layout.cols + RhsLayout::kCols);
  RUY_DCHECK_EQ((end_col - start_col) % RhsLayout::kCols, 0);
#if RUY_OPT(FAT_KERNEL)
  kernel.Run(lhs, rhs, mul_params, start_row, start_col, end_row, end_col, dst);
#else
  for (int col = start_col; col < end_col; col += RhsLayout::kCols) {
    int block_end_col = std::min(col + RhsLayout::kCols, end_col);
    for (int row = start_row; row < end_row; row += LhsLayout::kCols) {
      int block_end_row = std::min(row + LhsLayout::kCols, end_row);
      kernel.Run(lhs, rhs, mul_params, row, col, block_end_row, block_end_col,
                 dst);
    }
  }
#endif
}

// Main entry point for kernels.
template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void RunKernel(Tuning tuning, const SidePair<PEMat>& src, void* mul_params,
               const SidePair<int>& start, const SidePair<int>& end,
               EMat* dst) {
  Mat<DstScalar> mdst = UneraseType<DstScalar>(*dst);
  RunKernelTyped<ThePath, LhsScalar, RhsScalar, DstScalar, MulParamsType>(
      tuning, UneraseType<LhsScalar>(src[Side::kLhs]),
      UneraseType<RhsScalar>(src[Side::kRhs]),
      *static_cast<const MulParamsType*>(mul_params), start[Side::kLhs],
      start[Side::kRhs], end[Side::kLhs], end[Side::kRhs], &mdst);
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct Kernel<Path::kStandardCpp, LhsScalar, RhsScalar, DstScalar,
              MulParamsType> {
  using AccumScalar = typename MulParamsType::AccumScalar;
  using LhsLayout = typename MulParamsType::StandardCppKernelLhsLayout;
  using RhsLayout = typename MulParamsType::StandardCppKernelRhsLayout;
  explicit Kernel(Tuning) {}
  void Run(const PMat<LhsScalar>& lhs, const PMat<RhsScalar>& rhs,
           const MulParamsType& mul_params, int start_row, int start_col,
           int end_row, int end_col, Mat<DstScalar>* dst) const {
    // See the comment in RunKernelTyped. end_row may be larger than
    // dst->layout.rows. It's the responsibility of the kernel to avoid
    // overrunning dst boundaries, which we do here by computing
    // clamped_end_row.
    int clamped_end_row = std::min(end_row, dst->layout.rows);
    int clamped_end_col = std::min(end_col, dst->layout.cols);
    RUY_DCHECK_LE(0, start_row);
    RUY_DCHECK_LE(start_row, clamped_end_row);
    RUY_DCHECK_LE(clamped_end_row, dst->layout.rows);
    RUY_DCHECK_LE(clamped_end_row, end_row);
    RUY_DCHECK_LE(end_row - clamped_end_row, LhsLayout::kCols);
    RUY_DCHECK_LE(0, start_col);
    RUY_DCHECK_LE(start_col, clamped_end_col);
    RUY_DCHECK_LE(clamped_end_col, dst->layout.cols);
    RUY_DCHECK_LE(clamped_end_col, end_col);
    RUY_DCHECK_LE(end_col - clamped_end_col, RhsLayout::kCols);
    profiler::ScopeLabel label("Kernel (Standard Cpp)");
    const int depth = lhs.layout.rows;
    for (int i = start_row; i < clamped_end_row; i++) {
      for (int j = start_col; j < clamped_end_col; j++) {
        using AccumScalar = typename MulParamsType::AccumScalar;
        AccumScalar accum = 0;
        for (int k = 0; k < depth; k++) {
          AccumScalar lhs_val = Element(lhs, k, i);
          AccumScalar rhs_val = Element(rhs, k, j);
          accum += lhs_val * rhs_val;
        }
        if (mul_params.bias()) {
          accum += mul_params.bias()[i];
        }
        if (lhs.zero_point) {
          accum -= lhs.zero_point * rhs.sums[j];
        }
        if (rhs.zero_point) {
          accum -= rhs.zero_point * lhs.sums[i];
        }
        if (lhs.zero_point && rhs.zero_point) {
          accum += lhs.zero_point * rhs.zero_point * depth;
        }
        ApplyMultiplier(mul_params, i, &accum);
        accum += dst->zero_point;
        accum = std::min<AccumScalar>(accum, mul_params.clamp_max());
        accum = std::max<AccumScalar>(accum, mul_params.clamp_min());
        *ElementPtr(dst, i, j) = static_cast<DstScalar>(accum);
      }
    }
  }
};

}  // namespace ruy

#endif  // RUY_RUY_KERNEL_H_
