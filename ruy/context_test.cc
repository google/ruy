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

#include "ruy/context.h"

#include "ruy/gtest_wrapper.h"
#include "ruy/path.h"
#include "ruy/prepacked_cache.h"

namespace ruy {
namespace {

TEST(ContextTest, ContextClassSanity) {
  Context context;
  EXPECT_EQ(context.last_taken_path(), Path::kNone);
  EXPECT_EQ(context.explicit_tuning(), Tuning::kAuto);
  EXPECT_EQ(&context.thread_pool(), context.mutable_thread_pool());
  EXPECT_NE(context.mutable_thread_pool(), nullptr);
  EXPECT_EQ(context.max_num_threads(), 1);
  EXPECT_EQ(&context.tracing(), context.mutable_tracing());
  EXPECT_EQ(context.cache_policy(), CachePolicy::kNoCache);
  context.set_last_taken_path(Path::kStandardCpp);
  context.set_explicit_tuning(Tuning::kOutOfOrder);
  context.set_max_num_threads(2);
  context.set_cache_policy(CachePolicy::kCacheLHSOnNarrowMul);
  EXPECT_EQ(context.last_taken_path(), Path::kStandardCpp);
  EXPECT_EQ(context.explicit_tuning(), Tuning::kOutOfOrder);
  EXPECT_EQ(context.max_num_threads(), 2);
  EXPECT_EQ(context.cache_policy(), CachePolicy::kCacheLHSOnNarrowMul);
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
