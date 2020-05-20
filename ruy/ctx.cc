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

#include "ruy/ctx.h"

#include "ruy/check_macros.h"
#include "ruy/ctx_impl.h"
#include "ruy/detect_arm.h"
#include "ruy/detect_x86.h"
#include "ruy/have_built_path_for.h"
#include "ruy/platform.h"
#include "ruy/prepacked_cache.h"

namespace ruy {

const CtxImpl& Ctx::impl() const { return static_cast<const CtxImpl&>(*this); }
CtxImpl* Ctx::mutable_impl() { return static_cast<CtxImpl*>(this); }

Path Ctx::last_used_path() const { return impl().last_used_path_; }
Tuning Ctx::explicit_tuning() const { return impl().explicit_tuning_; }
void Ctx::set_explicit_tuning(Tuning value) {
  mutable_impl()->explicit_tuning_ = value;
}
const ThreadPool& Ctx::thread_pool() const { return impl().thread_pool_; }
ThreadPool* Ctx::mutable_thread_pool() { return &mutable_impl()->thread_pool_; }
int Ctx::max_num_threads() const { return impl().max_num_threads_; }
void Ctx::set_max_num_threads(int value) {
  mutable_impl()->max_num_threads_ = value;
}

void Ctx::SetRuntimeEnabledPaths(Path paths) {
  mutable_impl()->runtime_enabled_paths_ = paths;
}

Path Ctx::GetRuntimeEnabledPaths() {
  // Just a shorthand alias. Using a pointer to make it clear we're mutating
  // this value in-place.
  Path* paths = &mutable_impl()->runtime_enabled_paths_;

  // The value Path::kNone indicates the initial state before detection has been
  // performed.
  if (*paths == Path::kNone) {
    *paths = DetectRuntimeSupportedPaths(kAllPaths);
  }

  return *paths;
}

Path Ctx::SelectPath(Path compiled_paths) {
  return mutable_impl()->last_used_path_ =
             GetMostSignificantPath(compiled_paths & GetRuntimeEnabledPaths());
}

void Ctx::EnsureThreadSpecificResources(int thread_count) {
  auto& resources = mutable_impl()->thread_specific_resources_;
  while (thread_count > static_cast<int>(resources.size())) {
    resources.emplace_back(new ThreadSpecificResource);
  }
  RUY_DCHECK_LE(thread_count, static_cast<int>(resources.size()));
}

TuningResolver* Ctx::GetThreadSpecificTuningResolver(int thread_index) const {
  const auto& resources = impl().thread_specific_resources_;
  RUY_DCHECK_LT(thread_index, static_cast<int>(resources.size()));
  return &resources[thread_index]->tuning_resolver;
}

Allocator* Ctx::GetThreadSpecificAllocator(int thread_index) const {
  const auto& resources = impl().thread_specific_resources_;
  RUY_DCHECK_LT(thread_index, static_cast<int>(resources.size()));
  return &resources[thread_index]->allocator;
}

Allocator* Ctx::GetMainAllocator() {
  if (!impl().main_allocator_) {
    mutable_impl()->main_allocator_.reset(new Allocator);
  }
  return impl().main_allocator_.get();
}

PrepackedCache* Ctx::GetPrepackedCache() {
  if (!impl().prepacked_cache_) {
    mutable_impl()->prepacked_cache_.reset(new PrepackedCache);
  }
  return impl().prepacked_cache_.get();
}

Tuning Ctx::GetMainThreadTuning() {
  EnsureThreadSpecificResources(1);
  TuningResolver* tuning_resolver = GetThreadSpecificTuningResolver(0);
  tuning_resolver->SetTuning(explicit_tuning());
  return tuning_resolver->Resolve();
}

void Ctx::ClearPrepackedCache() { mutable_impl()->prepacked_cache_ = nullptr; }

}  // namespace ruy
