#include <iostream>

#include "ruy.h"

int main() {
  ruy::Context context;

  {
    const float lhs_data[] = {1.0, 2.0, 3.0, 4.0};
    const float rhs_data[] = {1.0, 2.0, 3.0, 4.0};
    float dst_data[4];

    ruy::Matrix<float> lhs;
    ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
    lhs.set_data(lhs_data);
    ruy::Matrix<float> rhs;
    ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
    rhs.set_data(rhs_data);
    ruy::Matrix<float> dst;
    ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
    dst.set_data(dst_data);

    ruy::BasicSpec<float, float> spec;
    ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, &context, &dst);

    std::cout << "Float example:\n";
    std::cout << "LHS:\n" << lhs;
    std::cout << "RHS:\n" << rhs;
    std::cout << "Result:\n" << dst << "\n";
  }

  {
    const std::uint8_t lhs_data[] = {1, 2, 3, 4};
    const std::uint8_t rhs_data[] = {1, 2, 3, 4};
    std::uint8_t dst_data[4];

    ruy::Matrix<std::uint8_t> lhs;
    ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
    lhs.set_data(lhs_data);
    ruy::Matrix<std::uint8_t> rhs;
    ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
    rhs.set_data(rhs_data);
    ruy::Matrix<std::uint8_t> dst;
    ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
    dst.set_data(dst_data);

    ruy::BasicSpec<std::int32_t, std::uint8_t> spec;
    spec.multiplier_fixedpoint = 1 << 30;
    ruy::Mul<ruy::kAllPaths>(lhs, rhs, spec, &context, &dst);

    std::cout << "Quantized example (note the quantized multiplier 2^30 means "
                 "one half)\n";
    std::cout << "LHS:\n" << lhs;
    std::cout << "RHS:\n" << rhs;
    std::cout << "Result:\n" << dst << "\n";
  }
}
