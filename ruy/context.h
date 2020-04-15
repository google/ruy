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

#ifndef RUY_RUY_CONTEXT_H_
#define RUY_RUY_CONTEXT_H_

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
// TODO(b/154013439) move this structure to context_internal.h.
// Have Context store raw PerThreadState* pointers so it doesn't need
// a definition.
struct PerThreadState {
  const TuningResolver& get_tuning_resolver() const { return tuning_resolver; }
  TuningResolver* mutable_tuning_resolver() { return &tuning_resolver; }
  const Allocator& get_allocator() const { return allocator; }
  Allocator* mutable_allocator() { return &allocator; }

  // Each thread may be running on a different microarchitecture. For example,
  // some threads may be on big cores, while others are on little cores. Thus,
  // it's best for the tuning to be per-thread.
  TuningResolver tuning_resolver;
  // Each thread has its own local allocator.
  Allocator allocator;
};

// A Context holds runtime information used by Ruy. It holds runtime resources
// such as the workers thread pool and the allocator (which holds buffers for
// temporary data), as well as runtime options controlling which Paths are
// enabled (typically based on which instruction sets are detected) and how
// many threads to use.
struct Context final {
  Path get_last_taken_path() const { return last_taken_path; }
  void set_last_taken_path(Path value) { last_taken_path = value; }
  Tuning get_explicit_tuning() const { return explicit_tuning; }
  void set_explicit_tuning(Tuning value) { explicit_tuning = value; }
  // See comment on workers_pool: we wanted to rename it all along.
  const ThreadPool& get_thread_pool() const { return workers_pool; }
  ThreadPool* mutable_thread_pool() { return &workers_pool; }
  int get_max_num_threads() const { return max_num_threads; }
  void set_max_num_threads(int value) { max_num_threads = value; }
  const TracingContext& get_tracing() const { return tracing; }
  TracingContext* mutable_tracing() { return &tracing; }
  CachePolicy get_cache_policy() const { return cache_policy; }
  void set_cache_policy(CachePolicy value) { cache_policy = value; }

  Path last_taken_path = Path::kNone;
  Tuning explicit_tuning = Tuning::kAuto;
  // TODO(b/154013439) rename that thread_pool. Current name is gemmlowp legacy.
  // And make it a pointer so that context.h does not
  // need to #include "thread_pool.h".
  ThreadPool workers_pool;
  int max_num_threads = 1;
  // TODO(b/154013439) make `tracing` a pointer so that context.h does not
  // need to #include "trace.h".
  TracingContext tracing;
  CachePolicy cache_policy = CachePolicy::kNoCache;

  void ClearPrepackedCache() { prepacked_cache_ = nullptr; }

 private:
  // Allocator for main thread work before invoking the threadpool.
  // Our simple Allocator does not allow reserving/allocating more blocks
  // while it's already in committed state, so the main thread needs both
  // this allocator, and its per-thread allocator.
  // TODO(b/154013439) make that a raw Allocator* so that context.h does not
  // need to #include "allocator.h"
  std::unique_ptr<Allocator> main_allocator_;
  // TODO(b/154013439) make that a raw PrepackedCache* so that context.h does
  // not need to #include "prepacked_cache.h"
  std::unique_ptr<PrepackedCache> prepacked_cache_;
  Path runtime_enabled_paths_ = Path::kNone;

  // State for each thread in the thread pool. Entry 0 is the main thread.
  // Only used internally in TrMul, so this doesn't have public accessors,
  // instead we befriend TrMul.
  std::vector<std::unique_ptr<PerThreadState>> per_thread_states;

  friend class ContextInternal;
};

}  // end namespace ruy

#endif  // RUY_RUY_CONTEXT_H_
