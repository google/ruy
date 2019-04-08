// This is the only Ruy header that users should #include.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_RUY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_RUY_H_

#include "context.h"
#include "dispatch.h"
#include "matrix.h"
#include "spec.h"

namespace ruy {

// Performs a multiplication of matrices.  This is Ruy's only API entry point.
// Should be self-explanatory given the above documentation for each of Matrix,
// Spec and Context. See reference code in reference.h, with the caveat that
// that is reference code for transpose-multiply (TrMul) not just multiply;
// see the translation between the two in transpose_dispatch.h.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void Mul(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
         const Spec& spec, Context* context, Matrix<DstScalar>* dst) {
  MulDispatch<CompiledPaths, LhsScalar, RhsScalar, DstScalar, Spec> dispatch;
  dispatch.Mul(lhs, rhs, spec, context, dst);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_RUY_H_
