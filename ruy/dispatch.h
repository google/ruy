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

// This file implements the translation between Ruy's entry point (ruy::Mul) and
// the internal implementation of matrix multiplication.
//
// The primary elements of this dispatch are:
// - pick suitable gemm kernel and packing routines for the user-specified
// CompiledPaths based on the current CPU.
// - decide on the structure of the packed matrices needed by the internal
// implementation (see pack.h for more information on packing).
// - translate the Mul operation into TrMul (see trmul.h for why that is
// useful). This is done by changing the matrix Layout -- no matrix data is
// actually moved.
//
// This file also performs some checking of invariants to catch user errors.

#ifndef RUY_RUY_DISPATCH_H_
#define RUY_RUY_DISPATCH_H_

#include <algorithm>
#include <cstdint>
#include <limits>  // IWYU pragma: keep
#include <type_traits>

#include "ruy/check_macros.h"
#include "ruy/common.h"
#include "ruy/ctx.h"
#include "ruy/kernel.h"
#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/pack.h"
#include "ruy/pack_common.h"
#include "ruy/path.h"
#include "ruy/prepacked_cache.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/side_pair.h"
#include "ruy/size_util.h"
#include "ruy/trmul.h"
#include "ruy/trmul_params.h"

namespace ruy {

// If the MulParamsType's LayoutSupport covers only some special cases,
// this function enforces that the matrix multiplication at hand falls into
// that special case.
template <typename MulParamsType>
void EnforceLayoutSupport(const MatLayout& lhs_layout,
                          const MatLayout& rhs_layout,
                          const MatLayout& dst_layout) {
  if (MulParamsType::kLayoutSupport == LayoutSupport::kRCC) {
    RUY_DCHECK(IsRowMajor(lhs_layout));
    RUY_DCHECK(IsColMajor(rhs_layout));
    RUY_DCHECK(IsColMajor(dst_layout));
  }
}

template <typename Scalar>
bool IsSymmetricZeroPoint(Scalar zero_point) {
  return zero_point == SymmetricZeroPoint<Scalar>();
}

template <typename MulParamsType, typename Scalar>
void CheckZeroPoint(Scalar zero_point) {
  if (std::is_floating_point<Scalar>::value ||
      MulParamsType::kZeroPointSupport == ZeroPointSupport::kSymmetric) {
    RUY_DCHECK(IsSymmetricZeroPoint(zero_point));
  }
}

template <typename MulParamsType, typename LhsScalar, typename RhsScalar,
          typename DstScalar>
void EnforceZeroPointSupport(LhsScalar lhs_zero_point, RhsScalar rhs_zero_point,
                             DstScalar dst_zero_point) {
  // If the MulParamsType's ZeroPointSupport covers only some special cases,
  // this function enforces that the matrix multiplication at hand falls into
  // that special case.
  CheckZeroPoint<MulParamsType>(lhs_zero_point);
  CheckZeroPoint<MulParamsType>(rhs_zero_point);
  CheckZeroPoint<MulParamsType>(dst_zero_point);

  // Guard against the case when both LHS and RHS zero_point's are equal to
  // the minimum representable value. In that case, padding with zero_point
  // values will generate the bad case for fast int8 kernels on NEON
  // (pre-dotprod) which attempt to multiply-accumulate two pairs of int8
  // into a int16:  this is safe except in the bad case -128*-128 + -128*-128.
  // See b/131609283. This only affects the kNeon path but we ban this for all
  // paths in order for ruy to have the same supported parameter space
  // on all paths.
  RUY_DCHECK(lhs_zero_point != std::numeric_limits<LhsScalar>::lowest() ||
             rhs_zero_point != std::numeric_limits<RhsScalar>::lowest());
}

template <typename MulParamsType, typename DstScalar>
void EnforceDstSpecSupport(const MulParamsType& mul_params,
                           DstScalar dst_zero_point) {
  static_assert(
      std::is_same<typename MulParamsType::DstScalar, DstScalar>::value, "");
  if (!std::is_same<typename MulParamsType::DstScalar, std::int32_t>::value)
    return;

  // If user is looking for the raw accumulator, zero_point and all the other
  // dequantize fields don't make sense and should not be set.
  RUY_DCHECK_EQ(dst_zero_point, 0);
  RUY_DCHECK_EQ(mul_params.clamp_max(),
                std::numeric_limits<std::int32_t>::max());
  RUY_DCHECK_EQ(mul_params.clamp_min(),
                std::numeric_limits<std::int32_t>::min());
  RUY_DCHECK_EQ(mul_params.multiplier_fixedpoint(), 0);
  RUY_DCHECK_EQ(mul_params.multiplier_exponent(), 0);
  RUY_DCHECK_EQ(mul_params.multiplier_fixedpoint_perchannel(), nullptr);
  RUY_DCHECK_EQ(mul_params.multiplier_exponent_perchannel(), nullptr);
}

inline bool IsColMajorTrMul(const TrMulParams& params) {
  return IsColMajor(params.src[Side::kLhs].layout) &&
         IsColMajor(params.src[Side::kRhs].layout) &&
         IsColMajor(params.dst.layout);
}

inline void CreatePackedLayout(const MatLayout& src, const Type& scalar,
                               const KernelLayout& kernel_layout,
                               PMatLayout* packed) {
  packed->order = Order::kColMajor;
  packed->rows = round_up_pot(src.rows, kernel_layout.rows);
  packed->cols = round_up_pot(src.cols, kernel_layout.cols);
  packed->kernel = kernel_layout;
  int inner_size = packed->rows;
  if (RUY_OPT(AVOID_ALIASING)) {
    packed->stride =
        (inner_size * scalar.size) % 1024 ? inner_size : inner_size + 64;
  } else {
    packed->stride = inner_size;
  }
}

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

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void CreateTrMulParams(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                       const MulParamsType& mul_params, Mat<DstScalar>* dst,
                       Path the_path, TrMulParams* params) {
  // Fill in the fields we already know.
  params->src[Side::kLhs] = EraseType(lhs);
  params->src[Side::kRhs] = EraseType(rhs);
  params->dst = EraseType(*dst);
  params->mul_params = ToVoidPtr(&mul_params);

  // Create inner loops and packed matrices based on the Path.
  PopulateTrMulParamsAllCompiledPaths<CompiledPaths, LhsScalar, RhsScalar,
                                      DstScalar, MulParamsType>(the_path,
                                                                params);
}

// Returns true if the operand on the given side should use caching of the
// packed form. This may either be explicitly dictated by its cache_policy
// (if it is kNeverCache, the default, or kAlwaysCache), or it may depend
// on a heuristic decision based on the other operand's width. For example,
// in a matrix*vector product, for the LHS matrix operand, the other side is
// the RHS vector, with a width of 1, causing the packing of the LHS to be
// a large fraction of the overall work, so a heuristic would typically
// decide in favor of caching, if permitted at all by the cache_policy.
inline bool ShouldCache(const TrMulParams& params, Side side) {
  const CachePolicy cache_policy = params.src[side].cache_policy;
  // The width that matters is that of the other side, it is what determines
  // the amortization of the packing work done on the present side.
  const Side other_side = Other(side);
  const int other_width = params.src[other_side].layout.cols;
  const int other_kernel_width = params.packed[other_side].layout.kernel.cols;
  switch (cache_policy) {
    case CachePolicy::kNeverCache:
      return false;
    case CachePolicy::kAlwaysCache:
      return true;
    case CachePolicy::kCacheIfLargeSpeedup:
      // The condition (other_width <= other_kernel_width) means that the kernel
      // will traverse each value of the present side only once, meaning that
      // the overhead of the packing work will be maximal, hence maximally
      // worth caching.
      return (other_width <= other_kernel_width);
    case CachePolicy::kCacheIfSignificantSpeedup:
      // Variant of the heuristic used in the kCacheIfLargeSpeedup case. The
      // kernel will run on each value of the present side only a few times,
      // so packing overhead will be significant.
      return (other_width <= 4 * other_kernel_width);
    default:
      RUY_DCHECK(false);
      return false;
  }
}

inline void HandlePrepackedCaching(TrMulParams* params, Ctx* ctx) {
  for (Side side : {Side::kLhs, Side::kRhs}) {
    if (ShouldCache(*params, side)) {
      auto* cache = ctx->GetPrepackedCache();
      auto action = cache->Get(params->src[side].data, &params->packed[side]);
      if (action == PrepackedCache::Action::kInsertedNewEntry) {
        params->RunPack(side, ctx->GetMainThreadTuning(), 0,
                        params->packed[side].layout.cols);
      }
      params->is_prepacked[side] = true;
    }
  }
}

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void DispatchMul(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                 const MulParamsType& mul_params, Ctx* ctx,
                 Mat<DstScalar>* dst) {
  static_assert(CompiledPaths != Path::kNone, "Must compile at least one Path");
  static_assert((CompiledPaths & ~kAllPaths) == Path::kNone,
                "CompiledPaths must be a subset of ruy::kAllPaths");

  profiler::ScopeLabel mul_label("Mul");
  profiler::ScopeLabel shape_specific_label("matmul shape: %dx%dx%d",
                                            lhs.layout.rows, lhs.layout.cols,
                                            rhs.layout.cols);

  EnforceLayoutSupport<MulParamsType>(lhs.layout, rhs.layout, dst->layout);
  EnforceZeroPointSupport<MulParamsType>(lhs.zero_point, rhs.zero_point,
                                         dst->zero_point);
  EnforceDstSpecSupport<MulParamsType>(mul_params, dst->zero_point);

  // This should be a constant, for a given machine and CompiledPaths.
  // There is a back door to override it for testing, but in production it will
  // always be the "best" Path, i.e. the one with the newest SIMD instructions
  // available on the present machine.
  //
  // Unfortunately, it is not a *static* constant, since it depends on runtime
  // detection of the available SIMD instructions.
  const Path the_path = ctx->SelectPath(CompiledPaths);

  // As described in the comment at the top of this file, Ruy internally
  // converts Mul into TrMul. We handle that here.
  Mat<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);
  TrMulParams params;
  CreateTrMulParams<CompiledPaths>(transposed_lhs, rhs, mul_params, dst,
                                   the_path, &params);
  HandlePrepackedCaching(&params, ctx);
  TrMul(&params, ctx);
}

}  // namespace ruy

#endif  // RUY_RUY_DISPATCH_H_
