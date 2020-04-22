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
// This file is also factored to serve as a building block for the advanced API
// as well.
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
#include "ruy/context.h"
#include "ruy/context_friend.h"
#include "ruy/kernel.h"
#include "ruy/kernel_common.h"
#include "ruy/mat.h"
#include "ruy/matrix.h"
#include "ruy/mul_params.h"
#include "ruy/opt_set.h"
#include "ruy/pack.h"
#include "ruy/pack_common.h"
#include "ruy/path.h"
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
  if (RUY_OPT_ENABLED(RUY_OPT_AVOID_ALIASING)) {
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
  static_assert((ThePath & Path::kReference) == Path::kNone,
                "Path::kReference should not do TrMul");
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

  params->local_data_cache_size = MulParamsType::local_data_cache_size();
  params->shared_data_cache_size = MulParamsType::shared_data_cache_size();

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

  return;
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

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
void ReferenceMul(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                  const MulParamsType& mul_params, Mat<DstScalar>* dst) {
  profiler::ScopeLabel label("ReferenceMul");
  for (int i = 0; i < lhs.layout.rows; i++) {
    for (int j = 0; j < rhs.layout.cols; j++) {
      using AccumScalar = typename MulParamsType::AccumScalar;
      AccumScalar accum = 0;
      for (int k = 0; k < lhs.layout.cols; k++) {
        AccumScalar lhs_val = Element(lhs, i, k);
        AccumScalar rhs_val = Element(rhs, k, j);
        accum += (lhs_val - lhs.zero_point) * (rhs_val - rhs.zero_point);
      }
      if (mul_params.bias()) {
        accum += mul_params.bias()[i];
      }
      ApplyMultiplier(mul_params, i, &accum);
      accum += dst->zero_point;
      accum = std::min<AccumScalar>(accum, mul_params.clamp_max());
      accum = std::max<AccumScalar>(accum, mul_params.clamp_min());
      *ElementPtr(dst, i, j) = static_cast<DstScalar>(accum);
    }
  }
}

// Compile-time dispatch to ReferenceMul. This allows us to statically ensure
// that there is no call to ReferenceMul in the user's binary.
template <bool ReferenceMulIsEnabled>
struct CompileTimeEnabledReferenceMul {
  template <typename LhsScalar, typename RhsScalar, typename DstScalar,
            typename MulParamsType>
  static void Run(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                  const MulParamsType& mul_params, Mat<DstScalar>* dst) {
    ReferenceMul(lhs, rhs, mul_params, dst);
  }
};

// When this partial specialization is chosen, it ensures that ReferenceMul
// is never compiled.
template <>
struct CompileTimeEnabledReferenceMul</*ReferenceMulIsEnabled=*/false> {
  template <typename LhsScalar, typename RhsScalar, typename DstScalar,
            typename MulParamsType>
  static void Run(const Mat<LhsScalar>&, const Mat<RhsScalar>&,
                  const MulParamsType&, Mat<DstScalar>*) {
    RUY_DCHECK(false);
  }
};

inline void HandlePrepackedCaching(TrMulParams* params,
                                   const SidePair<bool>& cacheable,
                                   Context* context) {
  if (context->cache_policy() == CachePolicy::kNoCache) {
    return;
  }

  if (context->cache_policy() == CachePolicy::kCacheLHSOnNarrowMul) {
    // TODO(b/149304278) Cache on dst.cols <= selected kernel width.
    if (!cacheable[Side::kLhs] || params->dst.layout.cols > 4) {
      return;
    }
    PrepackedCache* prepacked_cache = ContextFriend::GetPrepackedCache(context);
    auto cache_key = std::make_pair(reinterpret_cast<void*>(params->run_kernel),
                                    params->src[Side::kLhs].data);
    auto it = prepacked_cache->FindAndUpdate(cache_key);
    if (it != prepacked_cache->cend()) {
      params->packed[Side::kLhs].data = it->second.first.data;
      params->packed[Side::kLhs].sums = it->second.first.sums;
      params->is_prepacked[Side::kLhs] = true;
      return;
    }

    // Allocate the prepacked matrix.
    PrepackedMatrix prepacked_lhs;
    prepacked_lhs.data_size = DataSize(params->packed[Side::kLhs]);
    prepacked_lhs.sums_size = SumsSize(params->packed[Side::kLhs]);
    prepacked_cache->AllocatePrepackedMatrix(&prepacked_lhs);
    params->packed[Side::kLhs].data = prepacked_lhs.data;
    params->packed[Side::kLhs].sums = prepacked_lhs.sums;
    params->is_prepacked[Side::kLhs] = true;
    Tuning tuning = ContextFriend::GetMainThreadTuning(context);
    params->RunPack(Side::kLhs, tuning, 0,
                    params->packed[Side::kLhs].layout.cols);
    prepacked_cache->Insert(cache_key, prepacked_lhs);
    return;
  }
}

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void DispatchMul(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                 const MulParamsType& mul_params, Context* context,
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
  // always be the "best" Path. I.e. the one with the newest SIMD instructions
  // available on the present machine, and avoiding Path::kReference unless
  // no other path is compiled.
  //
  // Unfortunately, it is not a *static* constant, since it depends on runtime
  // detection of the available SIMD instructions.
  Path the_path = ContextFriend::SelectPath(context, CompiledPaths);

  // Production code should probably never execute Path::kReference.
  // Path::kReference implements a Mul, not a TrMul like the rest of Ruy, so if
  // that's what we need to do, then get it out of the way before going down the
  // TrMul path.
  if (the_path == Path::kReference) {
    constexpr bool ReferenceMulIsEnabled =
        (CompiledPaths & Path::kReference) != Path::kNone;
    CompileTimeEnabledReferenceMul<ReferenceMulIsEnabled>::Run(lhs, rhs,
                                                               mul_params, dst);
    return;
  }

  // As described in the comment at the top of this file, Ruy internally
  // converts Mul into TrMul. We handle that here.
  //
  // This is Ruy's main code path.
  constexpr Path TrMulCompiledPaths = CompiledPaths & ~Path::kReference;
  Mat<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);
  TrMulParams params;
  CreateTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, mul_params, dst,
                                        the_path, &params);
  SidePair<bool> cacheable(lhs.cacheable, rhs.cacheable);
  HandlePrepackedCaching(&params, cacheable, context);
  TrMul(&params, context);
}

}  // namespace ruy

#endif  // RUY_RUY_DISPATCH_H_
