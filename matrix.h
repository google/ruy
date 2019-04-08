#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_

#include <cstdint>
#include <type_traits>

#include "check_macros.h"

namespace ruy {

// Layout storage order. Here and elsewhere, 'col' is short for 'column'.
// 'column-major' means that each column is contiguous in memory.
enum class Order : std::uint8_t { kColMajor, kRowMajor };

struct KernelLayout final {
  Order order = Order::kColMajor;
  std::uint8_t rows = 1;
  std::uint8_t cols = 1;
};

// Describes the shape and storage layout of a matrix.
struct Layout final {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  // Stride is the offset between two adjacent matrix elements
  // in the non-contiguous direction.
  std::int32_t stride = 0;
  Order order = Order::kColMajor;

  KernelLayout kernel;
};

// A Matrix is really what Eigen and gemmlowp would have called a 'matrix map':
// it merely wraps existing data as a matrix. It doesn't own any buffer.
// Scalar may be any floating-point or integral type. When integral, it may be
// signed or unsigned.
template <typename Scalar>
struct Matrix final {
  static_assert(!std::is_const<Scalar>::value, "");

  void operator=(const Matrix& other) {
    mutable_data_ = other.mutable_data_;
#ifndef NDEBUG
    has_mutable_data_ = other.has_mutable_data_;
#endif
    layout = other.layout;
    zero_point = other.zero_point;
  }

 private:
  // Our take on the c++ problem of enforcing constness of data wrapped in a
  // containers class: it's not worth the hassle
  // of trying to make it fully work at compile-time. Instead, we only enforce
  // constness at runtime, and to make it zero-overhead, we only enforce it
  // in debug builds.
  union {
    Scalar* mutable_data_ = nullptr;
    const Scalar* const_data_;
  };
#ifndef NDEBUG
  bool has_mutable_data_ = false;
#endif

 public:
  void set_data(Scalar* data) {
    mutable_data_ = data;
#ifndef NDEBUG
    has_mutable_data_ = true;
#endif
  }
  void set_data(const Scalar* data) {
    const_data_ = data;
#ifndef NDEBUG
    has_mutable_data_ = false;
#endif
  }
  Scalar* data() {
#ifndef NDEBUG
    RUY_DCHECK(has_mutable_data_);
#endif
    return mutable_data_;
  }
  const Scalar* data() const { return const_data_; }

  // The shape and data layout of this matrix.
  Layout layout;
  // The zero_point, i.e. which Scalar value is to be interpreted as zero.
  // When Scalar is floating-point, this must be 0.
  Scalar zero_point = 0;
};

template <typename StreamType, typename Scalar>
StreamType& operator<<(StreamType& stream, const Matrix<Scalar>& mat) {
  for (int row = 0; row < mat.layout.rows; row++) {
    for (int col = 0; col < mat.layout.cols; col++) {
      stream << static_cast<double>(Element(mat, row, col)) << " ";
    }
    stream << "\n";
  }
  return stream;
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_
