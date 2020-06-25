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
#include "ruy/kernel.h"
#include "ruy/mat.h"
#include "ruy/mul_params.h"
#include "ruy/pack.h"
#include "ruy/path.h"
#include "ruy/trmul_params.h"

namespace ruy {
// While the only entry point to this file is CreateTrMulParams, its templatized
// nature requires putting more code in this header than we would like. This
// internal implementation code is enclosed in namespace 'detail'.
namespace detail {

void CreatePackedLayout(const MatLayout& src, const Type& scalar,
                        const KernelLayout& kernel_layout,
                        PMatLayout* packed_layout);

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
  PEMat* packed_matrix = &params->packed_matrix[side];
  packed_matrix->data_type = Type::Create<PackedScalar>();
  packed_matrix->sums_type = Type::Create<SumsType>();
  CreatePackedLayout(src.layout, packed_matrix->data_type, kernel_layout,
                     &packed_matrix->layout);
  packed_matrix->zero_point = Pack<PackedScalar, Scalar>(src.zero_point);
}

inline bool IsColMajorTrMul(const TrMulParams& params) {
  return IsColMajor(params.src[Side::kLhs].layout) &&
         IsColMajor(params.src[Side::kRhs].layout) &&
         IsColMajor(params.dst.layout);
}

template <typename KernelType>
struct CheckKernelPathImpl {
  static void Run(Path) {
    // Do nothing.
    // Path fallbacks are normal in general (see RUY_INHERIT_KERNEL).
    // That is to say that one may instantiate ruy::Mul with a weird combination
    // of types, such as LhsScalar==float and RhsScalar==double, and have it
    // work by silently falling back to Path::kStandardCpp. Only in specific
    // cases do we have dedicated kernels overriding that fallback, and that is
    // what partial specializations of this template will check.
  }
};

#if RUY_DCHECK_IS_ENABLED
template <Path ThePath, typename SrcScalar, typename AccumScalar,
          typename DstScalar>
struct CheckKernelPathImpl<Kernel<ThePath, SrcScalar, SrcScalar, DstScalar,
                                  MulParams<AccumScalar, DstScalar>>>
    final {
  using KernelType = Kernel<ThePath, SrcScalar, SrcScalar, DstScalar,
                            MulParams<AccumScalar, DstScalar>>;
  static void Run(Path expected_path) {
    // We want to assert that we are using a dedicated Kernel specialization and
    // not a fallback when we know we are in a case where such a kernel
    // specialization exists. At the moment in the current state of ruy's
    // architecture support for ARM and x86, that is when LhsScalar==RhsScalar
    // (already implied in this partial specialization) and when that type is
    // either float, int8, or uint8. Indeed, we have kernels supporting float
    // and int8, and we have the packing code converting uint8 to int8 (see
    // PackedTypeImpl).
    static constexpr bool kSrcScalarTypeSupportsFastKernels =
        std::is_same<SrcScalar, float>::value ||
        std::is_same<SrcScalar, std::int8_t>::value ||
        std::is_same<SrcScalar, std::uint8_t>::value;
    if (kSrcScalarTypeSupportsFastKernels) {
      RUY_DCHECK_EQ(expected_path, KernelType::kPath);
    }
  }
};
#endif

template <typename KernelType>
void CheckKernelPath(Path expected_path) {
  CheckKernelPathImpl<KernelType>::Run(expected_path);
}

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename AccumScalar, typename DstScalar>
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
    PopulateTrMulParams<Path::kStandardCpp, LhsScalar, RhsScalar, AccumScalar,
                        DstScalar>(params);
    return;
  }

  using PackedLhsScalar = PackedType<ThePath, LhsScalar>;
  using PackedRhsScalar = PackedType<ThePath, RhsScalar>;
  using Kernel =
      Kernel<ThePath, PackedLhsScalar, PackedRhsScalar, AccumScalar, DstScalar>;
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
  params->run_kernel = &RunKernel<Kernel>::Run;
  CheckKernelPath<Kernel>(ThePath);
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
          typename RhsScalar, typename AccumScalar, typename DstScalar>
struct PathSearchCountdown;

template <Path CompiledPaths, bool InCompiledPaths, int BitNumber,
          typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct PathSearchOnlyCompiledPaths {
  static constexpr Path kCurrentPath = static_cast<Path>(1 << BitNumber);
  static void Search(Path the_path, TrMulParams* params) {
    if (kCurrentPath == the_path) {
      PopulateTrMulParams<kCurrentPath, LhsScalar, RhsScalar, AccumScalar,
                          DstScalar>(params);
      return;
    }
    PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar, RhsScalar,
                        AccumScalar, DstScalar>::Search(the_path, params);
  }
};

// Skip this iteration if CompiledPaths doesn't contain the specified path.
template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename AccumScalar, typename DstScalar>
struct PathSearchOnlyCompiledPaths<CompiledPaths, false, BitNumber, LhsScalar,
                                   RhsScalar, AccumScalar, DstScalar> {
  static void Search(Path the_path, TrMulParams* params) {
    PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar, RhsScalar,
                        AccumScalar, DstScalar>::Search(the_path, params);
  }
};

template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename AccumScalar, typename DstScalar>
struct PathSearchCountdown {
  static constexpr Path kCurrentPath = static_cast<Path>(1 << BitNumber);
  static void Search(Path the_path, TrMulParams* params) {
    PathSearchOnlyCompiledPaths<
        CompiledPaths, (CompiledPaths & kCurrentPath) != Path::kNone, BitNumber,
        LhsScalar, RhsScalar, AccumScalar, DstScalar>::Search(the_path, params);
  }
};

// Termination of the countdown. If the counter reaches -1, then we haven't
// found the specified path.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename AccumScalar, typename DstScalar>
struct PathSearchCountdown<CompiledPaths, -1, LhsScalar, RhsScalar, AccumScalar,
                           DstScalar> {
  static void Search(Path, TrMulParams*) { RUY_DCHECK(false); }
};

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename AccumScalar, typename DstScalar>
void PopulateTrMulParamsAllCompiledPaths(Path the_path, TrMulParams* params) {
  return PathSearchCountdown<CompiledPaths, 8 * sizeof(Path) - 1, LhsScalar,
                             RhsScalar, AccumScalar,
                             DstScalar>::Search(the_path, params);
}

// Copy the underlying bytes of `mul_params` to `dst`, except that the specified
// `channel_dimension` value overrides the channel_dimension member in
// mul_params. The reason why channel_dimension is being special-cased among
// MulParams members is that we will need to transpose MulParams, and that
// consists just in toggling channel_dimension. In the typical envisioned usage
// pattern, mul_params is constant, so we cannot conveniently toggle its
// channel_dimension in place, and it would be slightly unfortunate to have to
// perform another copy of mul_params just for that.
template <typename AccumScalar, typename DstScalar>
void StoreMulParams(const MulParams<AccumScalar, DstScalar>& mul_params,
                    ChannelDimension channel_dimension, void* dst) {
  using MulParamsType = MulParams<AccumScalar, DstScalar>;
  static_assert(alignof(MulParamsType) <= kMaxMulParamsAlignment, "");
  static_assert(sizeof(MulParamsType) <= kMaxMulParamsSize, "");
  static_assert(std::is_trivially_copyable<MulParamsType>::value, "");
  std::memcpy(dst, &mul_params, sizeof(MulParamsType));
  static_cast<MulParamsType*>(dst)->set_channel_dimension(channel_dimension);
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
          typename AccumScalar, typename DstScalar>
void CreateTrMulParams(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                       const Mat<DstScalar>& dst,
                       const MulParams<AccumScalar, DstScalar>& mul_params,
                       Path the_path, TrMulParams* params) {
  // Fill in the fields we already know.
  params->src[Side::kLhs] = EraseType(lhs);
  params->src[Side::kRhs] = EraseType(rhs);
  params->dst = EraseType(dst);
  detail::StoreMulParams(mul_params, mul_params.channel_dimension(),
                         params->mul_params_bytes);

  // Create inner loops and packed matrices based on the Path.
  detail::PopulateTrMulParamsAllCompiledPaths<
      CompiledPaths, LhsScalar, RhsScalar, AccumScalar, DstScalar>(the_path,
                                                                   params);
}

}  // namespace ruy

#endif  // RUY_RUY_CREATE_TRMUL_PARAMS_H_
