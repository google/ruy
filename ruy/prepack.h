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

// Implementation of low-level pre-packing API.

#ifndef RUY_RUY_PREPACK_H_
#define RUY_RUY_PREPACK_H_

#include <cstddef>
#include <functional>

#include "ruy/check_macros.h"
#include "ruy/context.h"
#include "ruy/dispatch.h"
#include "ruy/internal_matrix.h"
#include "ruy/matrix.h"
#include "ruy/path.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/side_pair.h"
#include "ruy/spec.h"
#include "ruy/trmul.h"
#include "ruy/trmul_params.h"
#include "ruy/tune.h"

namespace ruy {

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void PrePackForMulInternal(const Matrix<LhsScalar>& lhs,
                           const Matrix<RhsScalar>& rhs, const Spec& spec,
                           Context* context, Matrix<DstScalar>* dst,
                           SidePair<PrepackedMatrix*> prepacked,
                           std::function<void*(int)> alloc_fn) {
  profiler::ScopeLabel label("PrePackForMul");
  Path the_path = context->GetPathToTake<CompiledPaths>();
  RUY_CHECK_NE(the_path, Path::kReference);
  constexpr Path TrMulCompiledPaths = CompiledPaths & ~Path::kReference;
  Matrix<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);
  TrMulParams params;
  CreateTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, spec, context, dst,
                                        the_path, &params);

  const SidePair<int> origin{0, 0};
  const SidePair<int> rounded_dims{params.packed[Side::kLhs].layout.cols,
                                   params.packed[Side::kRhs].layout.cols};

  Tuning tuning = context->GetMainThreadTuning();
  for (Side side : {Side::kLhs, Side::kRhs}) {
    if (prepacked[side]) {
      prepacked[side]->data_size = DataSize(params.packed[side]);
      prepacked[side]->sums_size = SumsSize(params.packed[side]);
      prepacked[side]->data = alloc_fn(prepacked[side]->data_size);
      prepacked[side]->sums = alloc_fn(prepacked[side]->sums_size);
      params.packed[side].data = prepacked[side]->data;
      params.packed[side].sums = prepacked[side]->sums;
      params.RunPack(side, tuning, origin[side], rounded_dims[side]);
    }
  }
}

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void MulWithPrepackedInternal(const Matrix<LhsScalar>& lhs,
                              const Matrix<RhsScalar>& rhs, const Spec& spec,
                              Context* context, Matrix<DstScalar>* dst,
                              SidePair<PrepackedMatrix*> prepacked) {
  profiler::ScopeLabel label("MulWithPrepacked");

  EnforceLayoutSupport<Spec>(lhs.layout, rhs.layout, dst->layout);
  EnforceZeroPointSupport<Spec>(lhs.zero_point, rhs.zero_point,
                                dst->zero_point);

  Path the_path = context->GetPathToTake<CompiledPaths>();
  RUY_CHECK_NE(the_path, Path::kReference);
  constexpr Path TrMulCompiledPaths = CompiledPaths & ~Path::kReference;
  Matrix<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);
  TrMulParams params;
  CreateTrMulParams<TrMulCompiledPaths>(transposed_lhs, rhs, spec, context, dst,
                                        the_path, &params);

  for (Side side : {Side::kLhs, Side::kRhs}) {
    if (prepacked[side]) {
      params.packed[side].data = prepacked[side]->data;
      params.packed[side].sums = prepacked[side]->sums;
      params.is_prepacked[side] = true;
    }
  }

  TrMul(&params, context);
}

}  // namespace ruy

#endif  // RUY_RUY_PREPACK_H_
