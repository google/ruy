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

#include "ruy/spec.h"

#include <cstdint>
#include <type_traits>

#include "ruy/gtest_wrapper.h"

namespace ruy {
namespace {

TEST(SpecTest, SpecClassSanity) {
  using MulParamsType = MulParams<std::int32_t, std::int8_t>;
  static_assert(std::is_same<MulParamsType::AccumScalar, std::int32_t>::value,
                "");
  static_assert(std::is_same<MulParamsType::DstScalar, std::int8_t>::value, "");

  MulParamsType spec;
  EXPECT_EQ(spec.get_bias(), nullptr);
  EXPECT_EQ(spec.get_multiplier_fixedpoint(), 0);
  EXPECT_EQ(spec.get_multiplier_exponent(), 0);
  EXPECT_EQ(spec.get_multiplier_fixedpoint_perchannel(), nullptr);
  EXPECT_EQ(spec.get_multiplier_exponent_perchannel(), nullptr);
  EXPECT_EQ(spec.get_clamp_min(), -128);
  EXPECT_EQ(spec.get_clamp_max(), 127);
  std::int32_t bias_data[1];
  spec.set_bias(bias_data);
  spec.set_multiplier_fixedpoint(123);
  spec.set_multiplier_exponent(4);
  EXPECT_EQ(spec.get_bias(), bias_data);
  EXPECT_EQ(spec.get_multiplier_fixedpoint(), 123);
  EXPECT_EQ(spec.get_multiplier_exponent(), 4);
  spec.set_multiplier_fixedpoint(0);
  spec.set_multiplier_exponent(0);
  std::int32_t multiplier_fixedpoint_perchannel_data[1];
  int multiplier_exponent_perchannel_data[1];
  spec.set_multiplier_fixedpoint_perchannel(
      multiplier_fixedpoint_perchannel_data);
  spec.set_multiplier_exponent_perchannel(multiplier_exponent_perchannel_data);
  spec.set_clamp_min(-10);
  spec.set_clamp_max(10);
  EXPECT_EQ(spec.get_multiplier_fixedpoint(), 0);
  EXPECT_EQ(spec.get_multiplier_exponent(), 0);
  EXPECT_EQ(spec.get_multiplier_fixedpoint_perchannel(),
            multiplier_fixedpoint_perchannel_data);
  EXPECT_EQ(spec.get_multiplier_exponent_perchannel(),
            multiplier_exponent_perchannel_data);
  EXPECT_EQ(spec.get_clamp_min(), -10);
  EXPECT_EQ(spec.get_clamp_max(), 10);
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
