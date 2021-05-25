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

#include "ruy/system_aligned_alloc.h"

#include <cstdint>
#include <type_traits>

#include "ruy/check_macros.h"
#include "ruy/gtest_wrapper.h"

namespace ruy {
namespace detail {
namespace {

TEST(SystemAlignedAllocTest, SystemAlignedAllocTest) {
  for (std::ptrdiff_t size = 1; size < 10000; size++) {
    void* ptr = SystemAlignedAlloc(size);
    RUY_CHECK(ptr);
    RUY_CHECK(
        !(reinterpret_cast<std::uintptr_t>(ptr) % kMinimumBlockAlignment));
    SystemAlignedFree(ptr);
  }
}

}  // namespace
}  // namespace detail
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
