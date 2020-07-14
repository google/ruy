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

#include "ruy/create_trmul_params.h"

#include "ruy/mat.h"
#include "ruy/mul_params.h"
#include "ruy/path.h"
#include "ruy/platform.h"

namespace ruy {
namespace detail {

void CreatePackedLayout(const MatLayout& src, const KernelLayout& kernel_layout,
                        PMatLayout* packed_layout) {
  packed_layout->order = Order::kColMajor;
  packed_layout->rows = round_up_pot(src.rows, kernel_layout.rows);
  packed_layout->cols = round_up_pot(src.cols, kernel_layout.cols);
  packed_layout->stride = packed_layout->rows;
  packed_layout->kernel = kernel_layout;
}

bool FallBackToStandardCpp(Path path, const SidePair<EMat>& src,
                           ChannelDimension channel_dimension,
                           bool perchannel_multiplier) {
  // Non-architecture-specific paths, including internal test-only paths,
  // are currently just variants of kStandardCpp, supporting every case thus
  // not requiring a fallback. Not falling back preserves test coverage that
  // is enabled by these internal test-only paths.
  if ((path & kNonArchPathsIncludingInternalVariants) != Path::kNone) {
    return false;
  }

  // Supporting row-major LHS/RHS would require transposing blocks in the
  // packing code. This isn't implemented at the moment, so we fall back to
  // StandardCpp when that would be needed.
  if (!IsColMajor(src[Side::kLhs].layout) ||
      !IsColMajor(src[Side::kRhs].layout)) {
    return true;
  }

#if RUY_PLATFORM_NEON_64
  return false;
#endif

#if RUY_PLATFORM_NEON_32
  return false;
#endif

#if RUY_PLATFORM_X86
  if (src[Side::kLhs].data_type == Type::Create<float>() ||
      path == Path::kAvx2Fma || perchannel_multiplier == false) {
    return false;
  }
#endif

  // Ruy's optimized kernels currently only support the channel_dimension==kRow
  // case.
  if (channel_dimension != ChannelDimension::kRow) {
    return true;
  }

  (void)path;
  (void)perchannel_multiplier;
  return false;
}

}  // namespace detail
}  // namespace ruy
