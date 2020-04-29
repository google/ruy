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

#include "ruy/ctx.h"
#include "ruy/ctx_impl.h"
#include "ruy/path.h"
#include "ruy/prepacked_cache.h"
#include "ruy/thread_pool.h"
#include "ruy/trace.h"
#include "ruy/tune.h"

namespace ruy {

Context::Context() : impl_(new CtxImpl) {}
Context::~Context() { delete impl_; }

const Ctx& Context::ctx() const { return static_cast<const Ctx&>(*impl_); }
Ctx* Context::mutable_ctx() { return static_cast<Ctx*>(impl_); }

Path Context::last_selected_path() const { return ctx().last_selected_path(); }
Tuning Context::explicit_tuning() const { return ctx().explicit_tuning(); }
void Context::set_explicit_tuning(Tuning value) {
  mutable_ctx()->set_explicit_tuning(value);
}
const ThreadPool& Context::thread_pool() const { return ctx().thread_pool(); }
ThreadPool* Context::mutable_thread_pool() {
  return mutable_ctx()->mutable_thread_pool();
}
int Context::max_num_threads() const { return ctx().max_num_threads(); }
void Context::set_max_num_threads(int value) {
  mutable_ctx()->set_max_num_threads(value);
}
const TracingContext& Context::tracing() const { return ctx().tracing(); }
TracingContext* Context::mutable_tracing() {
  return mutable_ctx()->mutable_tracing();
}

void Context::ClearPrepackedCache() { mutable_ctx()->ClearPrepackedCache(); }

}  // namespace ruy
