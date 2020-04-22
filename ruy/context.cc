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

#include "ruy/context.h"

#include "ruy/context_private.h"
#include "ruy/path.h"
#include "ruy/prepacked_cache.h"
#include "ruy/thread_pool.h"
#include "ruy/trace.h"
#include "ruy/tune.h"

namespace ruy {

Context::Context() : private_(new ContextPrivate) {}
Context::~Context() { delete private_; }

Path Context::last_selected_path() const {
  return private_->last_selected_path_;
}
Tuning Context::explicit_tuning() const { return private_->explicit_tuning_; }
void Context::set_explicit_tuning(Tuning value) {
  private_->explicit_tuning_ = value;
}
const ThreadPool& Context::thread_pool() const {
  return private_->thread_pool_;
}
ThreadPool* Context::mutable_thread_pool() { return &private_->thread_pool_; }
int Context::max_num_threads() const { return private_->max_num_threads_; }
void Context::set_max_num_threads(int value) {
  private_->max_num_threads_ = value;
}
const TracingContext& Context::tracing() const { return private_->tracing_; }
TracingContext* Context::mutable_tracing() { return &private_->tracing_; }
CachePolicy Context::cache_policy() const { return private_->cache_policy_; }
void Context::set_cache_policy(CachePolicy value) {
  private_->cache_policy_ = value;
}

void Context::ClearPrepackedCache() { private_->prepacked_cache_ = nullptr; }

}  // namespace ruy
