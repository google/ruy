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

// Context is the user-facing context class.

#ifndef RUY_RUY_CONTEXT_H_
#define RUY_RUY_CONTEXT_H_

#include <cstdint>

namespace ruy {

class Ctx;
class CtxImpl;
class ThreadPool;
enum class Path : std::uint8_t;
enum class Tuning;

// A Context holds runtime information used by Ruy. It holds runtime resources
// such as the workers thread pool and the allocator (which holds buffers for
// temporary data), as well as runtime options controlling which Paths are
// enabled (typically based on which instruction sets are detected) and how
// many threads to use.
class Context final {
 public:
  Context();
  ~Context();

  Path last_used_path() const;
  Tuning explicit_tuning() const;
  void set_explicit_tuning(Tuning value);
  const ThreadPool& thread_pool() const;
  ThreadPool* mutable_thread_pool();
  int max_num_threads() const;
  void set_max_num_threads(int value);

  void ClearPrepackedCache();

 private:
  CtxImpl* const impl_;

  const Ctx& ctx() const;
  Ctx* mutable_ctx();

  friend const Ctx* get_ctx(const Context*);
  friend Ctx* get_ctx(Context*);

  // Disallow copy
  Context(const Context&) = delete;
};

}  // end namespace ruy

#endif  // RUY_RUY_CONTEXT_H_
