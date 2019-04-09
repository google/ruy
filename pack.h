#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_

#include <cstdint>

#include "third_party/gemmlowp/profiling/instrumentation.h"
#include "common.h"
#include "opt_set.h"
#include "tune.h"

namespace ruy {

template <Path ThePath, typename Scalar>
struct PackedTypeImpl {
  using Type = Scalar;
};

template <>
struct PackedTypeImpl<Path::kNeon, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kNeonDotprod, std::uint8_t> {
  using Type = std::int8_t;
};

template <Path ThePath, typename Scalar>
using PackedType = typename PackedTypeImpl<ThePath, Scalar>::Type;

template <typename PackedScalar, typename Scalar>
PackedScalar Pack(Scalar x) {
  return x - SymmetricZeroPoint<Scalar>() + SymmetricZeroPoint<PackedScalar>();
}

template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar, typename AccumScalar>
struct PackImpl {};

#define RUY_INHERIT_PACK(PARENT, CHILD)                                        \
  template <typename FixedKernelLayout, typename Scalar,                       \
            typename PackedScalar, typename AccumScalar>                       \
  struct PackImpl<CHILD, FixedKernelLayout, Scalar, PackedScalar, AccumScalar> \
      : PackImpl<PARENT, FixedKernelLayout, Scalar, PackedScalar,              \
                 AccumScalar> {};

template <typename FixedKernelLayout, typename Scalar, typename PackedScalar,
          typename AccumScalar>
struct PackImpl<Path::kStandardCpp, FixedKernelLayout, Scalar, PackedScalar,
                AccumScalar> {
  static void Run(Tuning, const Matrix<Scalar>& src_matrix,
                  Matrix<PackedScalar>* packed_matrix, int start_col,
                  int end_col, AccumScalar* sums) {
    gemmlowp::ScopedProfilingLabel label("Pack (generic)");
    RUY_DCHECK_EQ((end_col - start_col) % FixedKernelLayout::kCols, 0);
    AccumScalar* sums_ptr = sums ? sums + start_col : nullptr;
    for (int block_col = start_col; block_col < end_col;
         block_col += FixedKernelLayout::kCols) {
      for (int c = 0; c < FixedKernelLayout::kCols; c++) {
        int col = block_col + c;
        AccumScalar accum = 0;
        for (int block_row = 0; block_row < packed_matrix->layout.rows;
             block_row += FixedKernelLayout::kRows) {
          for (int r = 0; r < FixedKernelLayout::kRows; r++) {
            int row = block_row + r;
            PackedScalar packed_val;
            if (col < src_matrix.layout.cols && row < src_matrix.layout.rows) {
              packed_val = Pack<PackedScalar>(Element(src_matrix, row, col));
            } else {
              packed_val = packed_matrix->zero_point;
            }
            accum += packed_val;
            PackedScalar* block_ptr = packed_matrix->data() +
                                      FixedKernelLayout::kCols * block_row +
                                      packed_matrix->layout.stride * block_col;
            relaxed_atomic_store(block_ptr + FixedKernelLayout::kRows * c + r,
                                 packed_val);
          }
        }
        if (sums) {
          relaxed_atomic_store(sums_ptr++, accum);
        }
      }
    }
  }
};

RUY_INHERIT_PACK(Path::kStandardCpp, Path::kNeon)
RUY_INHERIT_PACK(Path::kNeon, Path::kNeonDotprod)

#if (defined __aarch64__) && (RUY_OPT_SET & RUY_OPT_ASM)

void Pack8bitNeonOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                            const void* src_ptr2, const void* src_ptr3,
                            int src_inc0, int src_inc1, int src_inc2,
                            int src_inc3, int src_rows, int src_zero_point,
                            std::int8_t* packed_ptr, int start_col, int end_col,
                            std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonInOrder(const void* src_ptr0, const void* src_ptr1,
                         const void* src_ptr2, const void* src_ptr3,
                         int src_inc0, int src_inc1, int src_inc2, int src_inc3,
                         int src_rows, int src_zero_point,
                         std::int8_t* packed_ptr, int start_col, int end_col,
                         std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonDotprodOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                                   const void* src_ptr2, const void* src_ptr3,
                                   int src_inc0, int src_inc1, int src_inc2,
                                   int src_inc3, int src_rows,
                                   int src_zero_point, std::int8_t* packed_ptr,
                                   int start_col, int end_col,
                                   std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonDotprodInOrder(const void* src_ptr0, const void* src_ptr1,
                                const void* src_ptr2, const void* src_ptr3,
                                int src_inc0, int src_inc1, int src_inc2,
                                int src_inc3, int src_rows, int src_zero_point,
                                std::int8_t* packed_ptr, int start_col,
                                int end_col, std::int32_t* sums_ptr,
                                int input_xor);

template <typename Scalar>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kColMajor, 16, 4>, Scalar,
                std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  Matrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col, std::int32_t* sums) {
    RUY_DCHECK(IsLinearColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      int src_inc0 = 16;
      int src_inc1 = 16;
      int src_inc2 = 16;
      int src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data() + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        Pack8bitNeonInOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      } else {
        Pack8bitNeonOutOfOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      }
    }
  }
};

template <typename Scalar>
struct PackImpl<Path::kNeonDotprod, FixedKernelLayout<Order::kRowMajor, 4, 8>,
                Scalar, std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  Matrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col, std::int32_t* sums) {
    RUY_DCHECK(IsLinearColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data() +
          packed_matrix->layout.stride * (block_col & ~7) +
          ((block_col & 4) * 4);
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        Pack8bitNeonDotprodInOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      } else {
        Pack8bitNeonDotprodOutOfOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      }
    }
  }
};

void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc0, int src_inc1, int src_inc2,
                             int src_inc3, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col);
void PackFloatNeonInOrder(const float* src_ptr0, const float* src_ptr1,
                          const float* src_ptr2, const float* src_ptr3,
                          int src_inc0, int src_inc1, int src_inc2,
                          int src_inc3, int src_rows, int src_zero_point,
                          float* packed_ptr, int start_col, int end_col);

template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 4, 8>, float,
                float, float> {
  static void Run(Tuning tuning, const Matrix<float>& src_matrix,
                  Matrix<float>* packed_matrix, int start_col, int end_col,
                  float*) {
    RUY_DCHECK(IsLinearColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr = packed_matrix->data() +
                          packed_matrix->layout.stride * (block_col & ~7) +
                          ((block_col & 4));
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        PackFloatNeonInOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0,
                             src_inc1, src_inc2, src_inc3,
                             src_matrix.layout.rows, src_matrix.zero_point,
                             packed_ptr, start_col, end_col);
      } else {
        PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                src_inc0, src_inc1, src_inc2, src_inc3,
                                src_matrix.layout.rows, src_matrix.zero_point,
                                packed_ptr, start_col, end_col);
      }
    }
  }
};

#endif  // (defined __aarch64__) && (RUY_OPT_SET & RUY_OPT_ASM)

template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar, typename AccumScalar>
void Pack(Tuning tuning, const Matrix<Scalar>& src_matrix,
          Matrix<PackedScalar>* packed_matrix, int start_col, int end_col,
          AccumScalar* sums) {
  PackImpl<ThePath, FixedKernelLayout, Scalar, PackedScalar, AccumScalar>::Run(
      tuning, src_matrix, packed_matrix, start_col, end_col, sums);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_
