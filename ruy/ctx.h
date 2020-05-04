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

// Ctx is the internal context interface class used by most of ruy's own code.
// It is subclassed by CtxImpl which provides the actual data members.

#ifndef RUY_RUY_CTX_H_
#define RUY_RUY_CTX_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "ruy/allocator.h"
#include "ruy/path.h"
#include "ruy/thread_pool.h"
#include "ruy/trace.h"
#include "ruy/tune.h"

namespace ruy {

class CtxImpl;
class ThreadPool;
class TracingContext;
class Allocator;
class TuningResolver;
class PrepackedCache;
enum class Path : std::uint8_t;
enum class Tuning;

// Ctx is the internal context class used throughout ruy code. Whereas Context
// is exposed to users, Ctx is internal to ruy. As many of ruy's internal
// headers, included by ruy public headers, need to use Ctx, it is important
// that it does not include definition of all the actual data members. This is
// solved by a variant of the 'pimpl' idiom, where instead of being implemented
// in the usual way with a pointer member, it is implemented in a subclass,
// CtxImpl.
class Ctx /* not final, subclassed by CtxImpl */ {
 public:
  Path last_selected_path() const;
  Tuning explicit_tuning() const;
  void set_explicit_tuning(Tuning value);
  const ThreadPool& thread_pool() const;
  ThreadPool* mutable_thread_pool();
  int max_num_threads() const;
  void set_max_num_threads(int value);
  const TracingContext& tracing() const;
  TracingContext* mutable_tracing();

  void SetRuntimeEnabledPaths(Path paths);
  Path GetRuntimeEnabledPaths();
  Path SelectPath(Path compiled_paths);
  void EnsureThreadSpecificResources(int thread_count);
  TuningResolver* GetThreadSpecificTuningResolver(int thread_index) const;
  Allocator* GetThreadSpecificAllocator(int thread_index) const;
  Allocator* GetMainAllocator();
  PrepackedCache* GetPrepackedCache();
  Tuning GetMainThreadTuning();
  void ClearPrepackedCache();

 private:
  // Downcast helpers.
  const CtxImpl& impl() const;
  CtxImpl* mutable_impl();
};

}  // namespace ruy

#endif  // RUY_RUY_CTX_H_
