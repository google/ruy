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

// As a matrix multiplication library, Ruy offers a Mul entry point, performing
// matrix multiplication. For implementation purposes, it is much nicer to
// be dealing with the transpose-and-multiply operation, doing
//   Destination = Transpose(LHS) * RHS
// Indeed, the latter is performing dot-products between the *columns* of LHS
// and the columns of RHS, whereas a plain matrix multiplication is performing
// dot-products between the *rows* of LHS and the columns of RHS.
// That is why TrMul is nicer to implement, allowing for a more symmetric
// treatment of LHS and RHS.

#ifndef RUY_RUY_TRMUL_H_
#define RUY_RUY_TRMUL_H_

#include "ruy/ctx.h"
#include "ruy/trmul_params.h"

namespace ruy {

struct ContextInternal;
void TrMul(TrMulParams* params, Ctx* ctx);

}  // namespace ruy

#endif  // RUY_RUY_TRMUL_H_
