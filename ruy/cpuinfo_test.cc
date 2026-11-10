/* Copyright 2026 Google LLC. All Rights Reserved.

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

#include "ruy/cpuinfo.h"

#include <cstdint>

#include "ruy/cpu_cache_params.h"
#include "ruy/gtest_wrapper.h"

#ifdef RUY_HAVE_CPUINFO
#include <cpuinfo.h>
#endif

namespace ruy {
namespace {

TEST(CpuInfoTest, OutOfBoundsL2CachePointerDoesNotCrash) {
#ifdef RUY_HAVE_CPUINFO
  ASSERT_TRUE(cpuinfo_initialize());
  ASSERT_GT(cpuinfo_get_processors_count(), 0u);

  auto* proc = const_cast<cpuinfo_processor*>(cpuinfo_get_processor(0));
  ASSERT_NE(proc, nullptr);
  const cpuinfo_cache* saved_l2 = proc->cache.l2;

  // Simulate `processor->cache.l2 = l2 + UINT32_MAX` when `l2_index` is left
  // at `UINT32_MAX` during `cpuinfo` topology initialization.
  const std::uintptr_t l2_base =
      reinterpret_cast<std::uintptr_t>(cpuinfo_get_l2_caches());
  proc->cache.l2 = reinterpret_cast<const cpuinfo_cache*>(
      l2_base +
      static_cast<std::uintptr_t>(0xFFFFFFFFu) * sizeof(cpuinfo_cache));

  CpuInfo cpuinfo;
  const CpuCacheParams& params = cpuinfo.CacheParams();
  proc->cache.l2 = saved_l2;

  EXPECT_GT(params.local_cache_size, 0);
  EXPECT_GE(params.last_level_cache_size, params.local_cache_size);
#else
  CpuInfo cpuinfo;
  const CpuCacheParams& params = cpuinfo.CacheParams();
  EXPECT_GT(params.local_cache_size, 0);
  EXPECT_GE(params.last_level_cache_size, params.local_cache_size);
#endif
}

}  // namespace
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
