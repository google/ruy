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

// Implementation of MulFrontEnd, the front-end part of ruy.
// This is what the ruy::Mul entry point calls, and this ends in a call to
// TrMul, at which point we enter the middle-end.
// The front-end work includes parameter validation (Validate), detemplatization
// and resolution of the specific code path to take (CreateTrMulParams), and
// any additional logic best done upfront before entering the middle-end
// (e.g. HandlePrepackedCaching).
// The call to CreateTrMulParams is an important watershed in this code's
// structure: code before it needs to be templatized like the ruy::Mul entry
// point, code after it is un-templatized.

#ifndef RUY_RUY_FRONTEND_H_
#define RUY_RUY_FRONTEND_H_

#include "ruy/create_trmul_params.h"
#include "ruy/ctx.h"
#include "ruy/path.h"
#include "ruy/profiler/instrumentation.h"
#include "ruy/trmul_params.h"
#include "ruy/validate.h"

namespace ruy {

// The first half of front-end work, up to the point where we have TrMulParams.
// In other words, this is the part of the front-end work that needs to be
// templatized like the entry point, and that performs the initial work that
// requires this templatization, and the de-templatization. The output of this
// function is the TrMulParams, which contain enough information to allow the
// un-templatized code to take over from there.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void MulFrontEndUpToCreateTrMulParams(const Mat<LhsScalar>& lhs,
                                      const Mat<RhsScalar>& rhs,
                                      const Mat<DstScalar>& dst,
                                      const MulParamsType& mul_params, Ctx* ctx,
                                      TrMulParams* params) {
  static_assert(CompiledPaths != Path::kNone, "Must compile at least one Path");
  static_assert((CompiledPaths & ~kAllPaths) == Path::kNone,
                "CompiledPaths must be a subset of ruy::kAllPaths");

  // Perform validation of parameters early so that failures are easier to map
  // to user errors. In particular, perform this validation before the
  // transposition.
  Validate(lhs, rhs, dst, mul_params);

  // See the comment at the top of trmul.h. Ruy internally converts Mul into
  // TrMul. We handle that here.
  Mat<LhsScalar> transposed_lhs(lhs);
  Transpose(&transposed_lhs);

  // Determine which exact Path we're going to take in this Mul call.
  // This is cheap because it's cached in `ctx`. In user scenarios this always
  // evaluates to the same value on a given machine with given `CompiledPaths`,
  // but could be invalidated by a call to Ctx::SetRuntimeEnabledPaths(), which
  // might be exposed publicly in Context in the future.
  const Path the_path = ctx->SelectPath(CompiledPaths);

  // De-templatize this Mul call by creating a TrMulParams structure.
  // This is also where the specific kernel and pack code paths corresponding to
  // `the_path` are selected, among all the code paths in `CompiledPaths`, and
  // recorded as function pointers in the TrMulParams.
  CreateTrMulParams<CompiledPaths>(transposed_lhs, rhs, dst, mul_params,
                                   the_path, params);
}

// The second part of the front-end work, starting from where we have freshly
// created TrMulParams, performing any remaining front-end work and entering the
// middle-end.
void MulFrontEndFromTrMulParams(Ctx* ctx, TrMulParams* params);

// Top-level function orchestrating the two halves of front-end work:
// before and after we have detemplatized the call by creating TrMulParams.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void MulFrontEnd(const Mat<LhsScalar>& lhs, const Mat<RhsScalar>& rhs,
                 const MulParamsType& mul_params, Ctx* ctx,
                 Mat<DstScalar>* dst) {
  profiler::ScopeLabel mul_label("Mul");
  profiler::ScopeLabel shape_specific_label("matmul shape: %dx%dx%d",
                                            lhs.layout.rows, lhs.layout.cols,
                                            rhs.layout.cols);
  TrMulParams params;
  MulFrontEndUpToCreateTrMulParams<CompiledPaths>(lhs, rhs, *dst, mul_params,
                                                  ctx, &params);
  MulFrontEndFromTrMulParams(ctx, &params);
}

}  // namespace ruy

#endif  // RUY_RUY_FRONTEND_H_
