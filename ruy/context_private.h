/* Copyright 2020 Google LLC. All Rights Reserved.

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

#ifndef THIRD_PARTY_RUY_RUY_CONTEXT_PRIVATE_H_
#define THIRD_PARTY_RUY_RUY_CONTEXT_PRIVATE_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "ruy/allocator.h"
#include "ruy/path.h"
#include "ruy/prepacked_cache.h"
#include "ruy/thread_pool.h"
#include "ruy/trace.h"
#include "ruy/tune.h"

namespace ruy {

// The state private to each Ruy thread.
struct PerThreadState final {
  // Each thread may be running on a different microarchitecture. For example,
  // some threads may be on big cores, while others are on little cores. Thus,
  // it's best for the tuning to be per-thread.
  TuningResolver tuning_resolver;
  // Each thread has its own local allocator.
  Allocator allocator;
};

class ContextPrivate final {
  Path last_selected_path_ = Path::kNone;
  Tuning explicit_tuning_ = Tuning::kAuto;
  ThreadPool thread_pool_;
  int max_num_threads_ = 1;
  TracingContext tracing_;
  CachePolicy cache_policy_ = CachePolicy::kNoCache;
  // Allocator for main thread work before invoking the threadpool.
  // Our simple Allocator does not allow reserving/allocating more blocks
  // while it's already in committed state, so the main thread needs both
  // this allocator, and its per-thread allocator.
  std::unique_ptr<Allocator> main_allocator_;
  std::unique_ptr<PrepackedCache> prepacked_cache_;
  Path runtime_enabled_paths_ = Path::kNone;
  // State for each thread in the thread pool. Entry 0 is the main thread.
  // Only used internally in TrMul, so this doesn't have public accessors,
  // instead we befriend TrMul.
  std::vector<std::unique_ptr<PerThreadState>> per_thread_states_;

  friend class Context;
  friend class ContextFriend;
};

}  // namespace ruy

#endif  // THIRD_PARTY_RUY_RUY_CONTEXT_PRIVATE_H_
