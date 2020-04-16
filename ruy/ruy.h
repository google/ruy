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

// This is the main Ruy public header (see also ruy_advanced.h).

#ifndef RUY_RUY_RUY_H_
#define RUY_RUY_RUY_H_

#include "ruy/context.h"
#include "ruy/dispatch.h"
#include "ruy/matrix.h"
#include "ruy/path.h"
#include "ruy/spec.h"

namespace ruy {

// Performs a multiplication of matrices, with some extra features for
// neural network applications. The basic operation is:
//
//   dst = lhs * rhs    // matrix multiplication
//
// The `spec` argument conveys additional parameters that are not naturally
// associated with lhs, rhs, dst. That includes typical neural network
// application domain specific features such as a bias-vector and clamp bounds,
// as well as integer quantization parameters.
//
// The `context` argument can be any ruy::Context object as long as no other
// thread is going to concurrently access that ruy::Context. The simplest
// correct (but not efficient) calling pattern is
//
//   ruy::Context context;
//   ruy::Mul(lhs, rhs, spec, &context, dst);
//
// However, creating and destroying a new context everytime is inefficient
// because it doesn't allow for resources to persist across ruy calls. Such
// resources may include heap allocations, a thread pool, and hardware detection
// results, and can be expensive to obtain. So the recommended usage pattern is
// more like this:
//
//   // Once during initialization:
//   ruy::Context* context = new ruy::Context;
//
//   // Many times
//   ruy::Mul(lhs, rhs, spec, context, dst);
//
// If multiple threads may concurrently be calling ruy::Mul, they must either
// use separate Contexts, or use a lock to ensure that no two threads are
// concurrently accessing the Context object. There is no lock inside Context,
// nothing is done to ensure reentrancy with shared Context objects.
//
// Ruy defaults to using only 1 thread. Multi-threading is always opted in to,
// by calling Context::set_max_num_threads() with an explicit thread count.
// If multiple threads may concurrently be calling ruy::Mul, it is advisable
// to set up their respective Context objects with set_max_num_threads so that
// the overall number of threads doesn't exceed the overall number of threads
// that the system can usefully execute concurrently
// (e.g. the number of CPU cores in typical scenarios). At least ruy forces
// each invocation to make an explicit decision here, there is no automatic
// detection of the best number of threads to use in ruy.
template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename MulParamsType>
void Mul(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
         const MulParamsType& spec, Context* context, Matrix<DstScalar>* dst) {
  DispatchMul<ruy::kAllPaths, LhsScalar, RhsScalar, DstScalar, MulParamsType>(
      lhs, rhs, spec, context, dst);
}

// Variant of ruy::Mul allowing to specify a custom OR-ed set of Path's to
// compile. See the comments in path.h for more details.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename MulParamsType>
void Mul(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
         const MulParamsType& spec, Context* context, Matrix<DstScalar>* dst) {
  DispatchMul<CompiledPaths, LhsScalar, RhsScalar, DstScalar, MulParamsType>(
      lhs, rhs, spec, context, dst);
}

}  // namespace ruy

#endif  // RUY_RUY_RUY_H_
