/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "ruy/context_friend.h"

#include "ruy/context.h"
#include "ruy/gtest_wrapper.h"
#include "ruy/path.h"
#include "ruy/platform.h"

namespace ruy {
namespace {

TEST(ContextFriendTest, EnabledPathsGeneral) {
  ruy::Context context;
  const auto ruy_paths = ContextFriend::GetRuntimeEnabledPaths(&context);
  const auto ruy_paths_repeat = ContextFriend::GetRuntimeEnabledPaths(&context);
  ASSERT_EQ(ruy_paths, ruy_paths_repeat);
  EXPECT_NE(ruy_paths, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kReference, Path::kReference);
  EXPECT_EQ(ruy_paths & Path::kStandardCpp, Path::kStandardCpp);
}

#if RUY_PLATFORM(X86)
TEST(ContextFriendTest, EnabledPathsX86) {
  ruy::Context context;
  ContextFriend::SetRuntimeEnabledPaths(
      &context, Path::kSse42 | Path::kAvx2 | Path::kAvx512 | Path::kAvxVnni);
  const auto ruy_paths = ContextFriend::GetRuntimeEnabledPaths(&context);
  EXPECT_EQ(ruy_paths & Path::kReference, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kStandardCpp, Path::kNone);
}
#endif  // RUY_PLATFORM(X86)

#if RUY_PLATFORM(ARM)
TEST(ContextFriendTest, EnabledPathsArm) {
  ruy::Context context;
  ContextFriend::SetRuntimeEnabledPaths(&context,
                                        Path::kNeon | Path::kNeonDotprod);
  const auto ruy_paths = ContextFriend::GetRuntimeEnabledPaths(&context);
  EXPECT_EQ(ruy_paths & Path::kReference, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kStandardCpp, Path::kNone);
  EXPECT_EQ(ruy_paths & Path::kNeon, Path::kNeon);
}
#endif  // RUY_PLATFORM(ARM)

TEST(ContextFriendTest, GetPerThreadStates) {
  ruy::Context context;
  for (int i = 1; i <= 4; i++) {
    const auto& per_thread_states =
        ContextFriend::GetPerThreadStates(&context, i);
    EXPECT_EQ(per_thread_states.size(), i);
    for (int j = 0; j < i; j++) {
      EXPECT_NE(&per_thread_states[j]->get_allocator(), nullptr);
      EXPECT_EQ(&per_thread_states[j]->get_allocator(),
                per_thread_states[j]->mutable_allocator());
      EXPECT_NE(&per_thread_states[j]->get_tuning_resolver(), nullptr);
      EXPECT_EQ(&per_thread_states[j]->get_tuning_resolver(),
                per_thread_states[j]->mutable_tuning_resolver());
    }
    // Calling with a smaller thread_count should not shrink the vector.
    EXPECT_EQ(ContextFriend::GetPerThreadStates(&context, 1).size(), i);
  }
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
