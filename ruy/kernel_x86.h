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

#ifndef RUY_RUY_KERNEL_X86_H_
#define RUY_RUY_KERNEL_X86_H_

#include <cstdint>

#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/path.h"
#include "ruy/platform.h"
#include "ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM_X86

RUY_INHERIT_KERNEL(Path::kStandardCpp, Path::kAvx2Fma)
RUY_INHERIT_KERNEL(Path::kAvx2Fma, Path::kAvx512)

void Kernel8bitAvx512(const KernelParams8bit<16, 16>& params);
void Kernel8bitAvx512SingleCol(const KernelParams8bit<16, 16>& params);

template <typename DstScalar>
struct Kernel<Path::kAvx512, std::int8_t, std::int8_t, std::int32_t, DstScalar> {
  static constexpr Path kPath = Path::kAvx512;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<std::int8_t>& lhs, const PMat<std::int8_t>& rhs,
           const MulParams<std::int32_t, DstScalar>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, mul_params, start_row, start_col, end_row,
                         end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      Kernel8bitAvx512SingleCol(params);
    } else {
      Kernel8bitAvx512(params);
    }
  }
};

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params);
void KernelFloatAvx512SingleCol(const KernelParamsFloat<16, 16>& param);

template <>
struct Kernel<Path::kAvx512, float, float, float, float> {
  static constexpr Path kPath = Path::kAvx512;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<float>& lhs, const PMat<float>& rhs,
           const MulParams<float, float>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, mul_params, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      KernelFloatAvx512SingleCol(params);
    } else {
      KernelFloatAvx512(params);
    }
  }
};

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params);
void Kernel8bitAvx2SingleCol(const KernelParams8bit<8, 8>& params);

template <typename DstScalar>
struct Kernel<Path::kAvx2Fma, std::int8_t, std::int8_t, std::int32_t,
              DstScalar> {
  static constexpr Path kPath = Path::kAvx2Fma;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<std::int8_t>& lhs, const PMat<std::int8_t>& rhs,
           const MulParams<std::int32_t, DstScalar>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, mul_params, start_row, start_col, end_row,
                         end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      Kernel8bitAvx2SingleCol(params);
    } else {
      Kernel8bitAvx2(params);
    }
  }
};

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params);
void KernelFloatAvx2SingleCol(const KernelParamsFloat<8, 8>& params);

template <>
struct Kernel<Path::kAvx2Fma, float, float, float, float> {
  static constexpr Path kPath = Path::kAvx2Fma;
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PMat<float>& lhs, const PMat<float>& rhs,
           const MulParams<float, float>& mul_params, int start_row,
           int start_col, int end_row, int end_col, Mat<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, mul_params, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (dst->layout.cols == 1 &&
        mul_params.channel_dimension() == ChannelDimension::kRow) {
      KernelFloatAvx2SingleCol(params);
    } else {
      KernelFloatAvx2(params);
    }
  }
};

#endif  // RUY_PLATFORM_X86

}  // namespace ruy

#endif  // RUY_RUY_KERNEL_X86_H_
