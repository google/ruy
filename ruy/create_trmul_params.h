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

// Implementation of CreateTrMulParams, see function comment.

#ifndef RUY_RUY_CREATE_TRMUL_PARAMS_H_
#define RUY_RUY_CREATE_TRMUL_PARAMS_H_

#include <type_traits>

#include "ruy/ctx.h"
#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/mul_params.h"
#include "ruy/pack_common.h"
#include "ruy/path.h"
#include "ruy/trmul_params.h"

namespace ruy {
// While the only entry point to this file is CreateTrMulParams, its templatized
// nature requires putting more code in this header than we would like. This
// internal implementation code is enclosed in namespace 'detail'.
namespace detail {

void CreatePackedLayout(const MatLayout& src, const Type& scalar,
                        const KernelLayout& kernel_layout, PMatLayout* packed);

template <typename Scalar, typename PackedScalar>
void CreatePackedMatrix(Side side, const KernelLayout& kernel_layout,
                        TrMulParams* params) {
  // Ruy always uses 32-bit signed accumulators for quantized
  // matrix multiplication, so we would like to always use std::int32_t
  // unconditionally for SumsType.
  // However, for floating point types, we still need a reasonable type here to
  // avoid tripping assertions elsewhere in the code.
  using SumsType =
      typename std::conditional<std::is_floating_point<Scalar>::value, Scalar,
                                std::int32_t>::type;

  const EMat& src = params->src[side];
  PEMat* packed = &params->packed[side];
  packed->data_type = Type::Create<PackedScalar>();
  packed->sums_type = Type::Create<SumsType>();
  CreatePackedLayout(src.layout, packed->data_type, kernel_layout,
                     &packed->layout);
  packed->zero_point = Pack<PackedScalar, Scalar>(src.zero_point);
}

inline bool IsColMajorTrMul(const TrMulParams& params) {
  return IsColMajor(params.src[Side::kLhs].layout) &&
         IsColMajor(params.src[Side::kRhs].layout) &&
         IsColMajor(params.dst.layout);
}

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void PopulateTrMulParams(TrMulParams* params) {
  // The optimized code paths don't handle the full generality of Ruy's API.
  // Fall back to Path::kStandardCpp if necessary.
  bool fallback_to_standard_cpp = false;
  if (ThePath != Path::kStandardCpp) {
    // The optimized code paths currently only handle the case of all matrices
    // being column major.
    if (!IsColMajorTrMul(*params)) {
      fallback_to_standard_cpp = true;
    }
  }

  if (fallback_to_standard_cpp) {
    PopulateTrMulParams<Path::kStandardCpp, LhsScalar, RhsScalar, DstScalar,
                        MulParamsType>(params);
    return;
  }

  using PackedLhsScalar = PackedType<ThePath, LhsScalar>;
  using PackedRhsScalar = PackedType<ThePath, RhsScalar>;
  using Kernel = Kernel<ThePath, PackedLhsScalar, PackedRhsScalar, DstScalar,
                        MulParamsType>;
  using LhsKernelLayout = typename Kernel::LhsLayout;
  using RhsKernelLayout = typename Kernel::RhsLayout;

  params->path = ThePath;

  CreatePackedMatrix<LhsScalar, PackedLhsScalar>(
      Side::kLhs, ToKernelLayout<LhsKernelLayout>(), params);
  CreatePackedMatrix<RhsScalar, PackedRhsScalar>(
      Side::kRhs, ToKernelLayout<RhsKernelLayout>(), params);
  params->run_pack[Side::kLhs] =
      &RunPack<ThePath, LhsKernelLayout, LhsScalar, PackedLhsScalar>;
  params->run_pack[Side::kRhs] =
      &RunPack<ThePath, RhsKernelLayout, RhsScalar, PackedRhsScalar>;
  params->run_kernel = &RunKernel<ThePath, PackedLhsScalar, PackedRhsScalar,
                                  DstScalar, MulParamsType>;
}

// PopulateTrMulParamsAllCompiledPaths calls into one of multiple
// instantiations of PopulateTrMulParams. For each bit that is set in
// CompiledPaths, it statically instantiates PopulateTrMulParams with a Path
// corresponding to that single bit. The call to PopulateTrMulParams is
// guarded by a runtime check that it is in fact the dynamically selected path.
//
// PopulateTrMulParamsAllCompiledPaths is implemented with template
// metaprogramming by mutual recursion between PathSearchCountdown and
// PathSearchCompiledPaths.
//
// PopulateTrMulParamsAllCompiledPaths is logically implementing the following
// computation:
//
// template <Path CompiledPaths>
// void PopulateTrMulParamsAllCompiledPaths(Path the_path,
//                                            TrMulParams* params) {
//   for (int bit = 8 * sizeof(Path) - 1; bit != -1; bit--) { // [1]
//     Path current_path = static_cast<Path>(1 << bit);
//     if ((CompiledPaths & current_path) != Path::kNone) { // [2]
//       if (current_path == the_path) { // [3]
//         PopulateTrMulParams<current_path, ...>(the_path, params);
//         return;
//       }
//     }
//   }
// }
//
//
//
// [1] - Done by the main definition of PathSearchCountdown. The `bit--` is
// done in the recursion of PathSearchOnlyCompiledPaths.
// [2] - Done by PathSearchOnlyCompiledPaths's partial template
// specialization on InCompiledPaths. This is the check which necessitates
// doing the whole computation at C++ compile time.
// [3] - Done by the `if` in the main definition of
// PathSearchOnlyCompiledPaths.
//
// The template metaprogramming is necessary because:
// - In `PopulateTrMulParams<current_path, ...>`, current_path must be a C++
// compile-time constant.
// - PopulateTrMulParamsAllCompiledPaths must not instantiate
// inner loops for paths that are not in CompiledPaths, since that can result in
// bogus instantiations which cause a compile time failure.
template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename MulParamsType>
struct PathSearchCountdown;

template <Path CompiledPaths, bool InCompiledPaths, int BitNumber,
          typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
struct PathSearchOnlyCompiledPaths {
  static constexpr Path kCurrentPath = static_cast<Path>(1 << BitNumber);
  static void Search(Path the_path, TrMulParams* params) {
    if (kCurrentPath == the_path) {
      PopulateTrMulParams<kCurrentPath, LhsScalar, RhsScalar, DstScalar,
                          MulParamsType>(params);
      return;
    }
    PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar, RhsScalar,
                        DstScalar, MulParamsType>::Search(the_path, params);
  }
};

// Skip this iteration if CompiledPaths doesn't contain the specified path.
template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename MulParamsType>
struct PathSearchOnlyCompiledPaths<CompiledPaths, false, BitNumber, LhsScalar,
                                   RhsScalar, DstScalar, MulParamsType> {
  static void Search(Path the_path, TrMulParams* params) {
    PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar, RhsScalar,
                        DstScalar, MulParamsType>::Search(the_path, params);
  }
};

template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename MulParamsType>
struct PathSearchCountdown {
  static constexpr Path kCurrentPath = static_cast<Path>(1 << BitNumber);
  static void Search(Path the_path, TrMulParams* params) {
    PathSearchOnlyCompiledPaths<
        CompiledPaths, (CompiledPaths & kCurrentPath) != Path::kNone, BitNumber,
        LhsScalar, RhsScalar, DstScalar, MulParamsType>::Search(the_path,
                                                                params);
  }
};

// Termination of the countdown. If the counter reaches -1, then we haven't
// found the specified path.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
struct PathSearchCountdown<CompiledPaths, -1, LhsScalar, RhsScalar, DstScalar,
                           MulParamsType> {
  static void Search(Path, TrMulParams*) { RUY_DCHECK(false); }
};

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void PopulateTrMulParamsAllCompiledPaths(Path the_path, TrMulParams* params) {
  return PathSearchCountdown<CompiledPaths, 8 * sizeof(Path) - 1, LhsScalar,
                             RhsScalar, DstScalar,
                             MulParamsType>::Search(the_path, params);
}

}  // namespace detail

// CreateTrMulParams is where we turn templatized ruy::Mul parameters to an
// ordinary, un-templatized runtime data structure, TrMulParams. This is, in
// particular, where we go from a template parameter 'CompiledPaths' describing
// the set of paths enabled at compile-time, to runtime parameter values
// corresponding to the specific choice of Path to be taken (encoded as function
// pointers to kernel and pack code paths, and as runtime values of their
// corresponding compile-time attributes e.g. the kernel-specific block layout).
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void CreateTrMulParams(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                       const Mat<DstScalar>& dst,
                       const MulParamsType& mul_params, Path the_path,
                       TrMulParams* params) {
  // Fill in the fields we already know.
  params->src[Side::kLhs] = EraseType(lhs);
  params->src[Side::kRhs] = EraseType(rhs);
  params->dst = EraseType(dst);
  params->mul_params = ToVoidPtr(&mul_params);

  // Create inner loops and packed matrices based on the Path.
  detail::PopulateTrMulParamsAllCompiledPaths<
      CompiledPaths, LhsScalar, RhsScalar, DstScalar, MulParamsType>(the_path,
                                                                     params);
}

}  // namespace ruy

#endif  // RUY_RUY_CREATE_TRMUL_PARAMS_H_
