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

#include "ruy/prepacked_cache.h"

#include <thread>  // NOLINT(build/c++11)

#include "ruy/context.h"
#include "ruy/context_get_ctx.h"
#include "ruy/gtest_wrapper.h"
#include "ruy/mat.h"
#include "ruy/ruy.h"
#include "ruy/time.h"

namespace ruy {
namespace {

PEMat MakeDummyPEMat(Type data_type, int rows, int cols) {
  PEMat ret;
  ret.data_type = data_type;
  if (!data_type.is_floating_point) {
    ret.sums_type = Type::Create<std::int32_t>();
  }
  ret.layout.rows = rows;
  ret.layout.cols = cols;
  ret.layout.stride = rows;
  ret.layout.order = Order::kColMajor;
  // The kernel block layout is not relevant to this test, so we leave it
  // trivial 1x1.
  ret.layout.kernel.rows = 1;
  ret.layout.kernel.cols = 1;
  return ret;
}

TEST(PrepackedCacheTest, TestCacheEjection) {
  // Create the cache.
  PrepackedCache prepacked_cache(306);
  // Allocate the prepacked matrix.
  // DataBytes=200, SumsBytes=20*4=80, Total: 280 bytes
  PEMat mat1 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 10, 20);
  prepacked_cache.AllocatePrepackedMatrix(&mat1);
  auto cache_key1 = std::make_pair(nullptr, mat1.data);
  prepacked_cache.Insert(cache_key1, mat1);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Get a time point after the insertion into the cache.
  TimePoint current = CoarseNow();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  PrepackedCache::CacheIterator itr = prepacked_cache.FindAndUpdate(cache_key1);
  EXPECT_NE(itr, prepacked_cache.cend());
  // By finding mat1, we updated its timestamp. Verify that `current` is older
  // than the time stamp now associated with mat1.
  EXPECT_LT(current, itr->second.second);
  // DataBytes=15, SumsBytes=3*4=12, Total: 27 bytes
  PEMat mat2 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 5, 3);
  prepacked_cache.AllocatePrepackedMatrix(&mat2);
  auto cache_key2 = std::make_pair(nullptr, mat2.data);
  prepacked_cache.Insert(cache_key2, mat2);
  // The cache size was exceeded by inserting mat2. Ensure that mat1 was
  // ejected.
  EXPECT_EQ(prepacked_cache.FindAndUpdate(cache_key1), prepacked_cache.cend());
}

TEST(PrepackedCacheTest, TestCacheBasic) {
  // Create the cache.
  PrepackedCache prepacked_cache(307);
  // Allocate the prepacked matrix.
  // DataBytes=200, SumsBytes=20*4=80, Total: 280 bytes
  PEMat mat1 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 10, 20);
  prepacked_cache.AllocatePrepackedMatrix(&mat1);
  auto cache_key1 = std::make_pair(nullptr, mat1.data);
  prepacked_cache.Insert(cache_key1, mat1);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key1), prepacked_cache.cend());
  // DataBytes=15, SumsBytes=3*4=12, Total: 27 bytes
  PEMat mat2 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 5, 3);
  prepacked_cache.AllocatePrepackedMatrix(&mat2);
  auto cache_key2 = std::make_pair(nullptr, mat2.data);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  prepacked_cache.Insert(cache_key2, mat2);
  // The cache size was not exceeded by inserting mat2. Ensure that mat1 was not
  // ejected.
  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key1), prepacked_cache.cend());
}

TEST(PrepackedCacheTest, TestCacheEjection2) {
  // Create the cache.
  PrepackedCache prepacked_cache(1000);
  // Allocate the prepacked matrix 1.
  // DataBytes=200, SumsBytes=20*4=80, Total: 280 bytes
  PEMat mat1 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 10, 20);
  prepacked_cache.AllocatePrepackedMatrix(&mat1);
  auto cache_key1 = std::make_pair(nullptr, mat1.data);
  prepacked_cache.Insert(cache_key1, mat1);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Allocate the prepacked matrix 2.
  // DataBytes=200, SumsBytes=20*4=80, Total: 280 bytes
  PEMat mat2 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 10, 20);
  prepacked_cache.AllocatePrepackedMatrix(&mat2);
  auto cache_key2 = std::make_pair(nullptr, mat2.data);
  prepacked_cache.Insert(cache_key2, mat2);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Allocate the prepacked matrix 3.
  // DataBytes=200, SumsBytes=20*4=80, Total: 280 bytes
  PEMat mat3 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 10, 20);
  prepacked_cache.AllocatePrepackedMatrix(&mat3);
  auto cache_key3 = std::make_pair(nullptr, mat3.data);
  prepacked_cache.Insert(cache_key3, mat3);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // The next insertion will cause the cache size to go over the ejection
  // threshold. Touch matrix 1 and matrix 3 to make matrix 2 the oldest
  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key1), prepacked_cache.cend());
  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key3), prepacked_cache.cend());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Allocate the prepacked matrix 4.
  // DataBytes=200, SumsBytes=20*4=80, Total: 280 bytes
  PEMat mat4 = MakeDummyPEMat(Type::Create<std::uint8_t>(), 10, 20);
  prepacked_cache.AllocatePrepackedMatrix(&mat4);
  auto cache_key4 = std::make_pair(nullptr, mat4.data);
  prepacked_cache.Insert(cache_key4, mat4);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Ensure that mat2 was ejected, but mat1, mat3, and mat4 were not.
  EXPECT_EQ(prepacked_cache.FindAndUpdate(cache_key2), prepacked_cache.cend());
  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key3), prepacked_cache.cend());
  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key1), prepacked_cache.cend());
  EXPECT_NE(prepacked_cache.FindAndUpdate(cache_key4), prepacked_cache.cend());
}

void TestCachePolicies(CachePolicy cache_policy, bool expected_cached) {
  ruy::Context context;
  ruy::Ctx* ctx = get_ctx(&context);
  PrepackedCache* cache = ctx->GetPrepackedCache();
  EXPECT_EQ(cache->TotalSize(), 0);

  const float lhs_data[] = {1, 2, 3, 4};
  const float rhs_data[] = {1, 2};
  float dst_data[4];

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, lhs.mutable_layout());
  lhs.set_data(lhs_data);
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 1, ruy::Order::kColMajor, rhs.mutable_layout());
  rhs.set_data(rhs_data);
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 1, ruy::Order::kColMajor, dst.mutable_layout());
  dst.set_data(dst_data);

  ruy::MulParams<float, float> mul_params;
  // Perform the multiplication and confirm no caching occurred.
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, mul_params, &context, &dst);
  EXPECT_EQ(cache->TotalSize(), 0);

  // Set cache policy for the LHS, repeat the multiplication, and see
  // that caching did occur.
  lhs.set_cache_policy(cache_policy);
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, mul_params, &context, &dst);
  const bool actual_cached = cache->TotalSize() > 0;
  EXPECT_EQ(actual_cached, expected_cached);
}

TEST(PrepackedCacheTest, TestCachePolicies) {
  for (CachePolicy cache_policy :
       {CachePolicy::kNeverCache, CachePolicy::kCacheIfLargeSpeedup,
        CachePolicy::kCacheIfSignificantSpeedup, CachePolicy::kAlwaysCache}) {
    TestCachePolicies(cache_policy,
                         cache_policy != CachePolicy::kNeverCache);
  }
}

TEST(PrepackedCacheTest, TestClearCache) {
  ruy::Context context;
  PrepackedCache* cache = get_ctx(&context)->GetPrepackedCache();
  EXPECT_EQ(cache->TotalSize(), 0);

  const float lhs_data[] = {1, 2, 3, 4};
  const float rhs_data[] = {1, 2};
  float dst_data[4];

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, lhs.mutable_layout());
  lhs.set_data(lhs_data);
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 1, ruy::Order::kColMajor, rhs.mutable_layout());
  rhs.set_data(rhs_data);
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 1, ruy::Order::kColMajor, dst.mutable_layout());
  dst.set_data(dst_data);

  ruy::MulParams<float, float> mul_params;
  // Set cache policy for the LHS and see that caching occurs.
  lhs.set_cache_policy(CachePolicy::kAlwaysCache);
  ruy::Mul<ruy::kAllPaths>(lhs, rhs, mul_params, &context, &dst);
  EXPECT_NE(cache->TotalSize(), 0);

  // Clear the cache via the Context.
  context.ClearPrepackedCache();
  // Verify that the cache is now empty.
  cache = get_ctx(&context)->GetPrepackedCache();
  EXPECT_EQ(cache->TotalSize(), 0);
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
