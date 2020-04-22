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

#include "ruy/context_internal.h"

#include "ruy/check_macros.h"
#include "ruy/context.h"
#include "ruy/detect_arm.h"
#include "ruy/detect_x86.h"
#include "ruy/have_built_path_for.h"
#include "ruy/platform.h"

namespace ruy {

void ContextInternal::SetRuntimeEnabledPaths(Context* context, Path paths) {
  context->runtime_enabled_paths_ = paths;
}

Path ContextInternal::GetRuntimeEnabledPaths(Context* context) {
  // Note, this is a reference, used to avoid exceedingly long lines in this
  // function body. Assigning to it mutates *context.
  Path& enabled_paths = context->runtime_enabled_paths_;

  // This function should always return the same value on a given machine.
  // When runtime_enabled_paths_ has its initial value kNone, it performs
  // some platform detection to resolve it to specific Path values.

  // Fast path: already resolved.
  if (enabled_paths != Path::kNone) {
    return enabled_paths;
  }

  // Need to resolve now. Start by considering all paths enabled.
  enabled_paths = kAllPaths;

  // This mechanism is intended to be used for testing and benchmarking. For
  // example, one can set RUY_FORCE_DISABLE_PATHS to Path::kAvx512 in order to
  // evaluate AVX2 performance on an AVX-512 machine.
#ifdef RUY_FORCE_DISABLE_PATHS
  enabled_paths = enabled_paths & ~(RUY_FORCE_DISABLE_PATHS);
#endif

#if RUY_PLATFORM(ARM)
  // Now selectively disable paths that aren't supported on this machine.
  if ((enabled_paths & Path::kNeonDotprod) != Path::kNone) {
    if (!DetectDotprod()) {
      enabled_paths = enabled_paths & ~Path::kNeonDotprod;
      // Sanity check.
      RUY_DCHECK((enabled_paths & Path::kNeonDotprod) == Path::kNone);
    }
  }
#endif  // RUY_PLATFORM(ARM)

#if RUY_PLATFORM(X86)
  // TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete /
  // placeholder. Optimization is not finished. In particular the dimensions of
  // the kernel blocks can be changed as desired.
  //
  if ((enabled_paths & Path::kSse42) != Path::kNone) {
    if (!(HaveBuiltPathForSse42() && DetectCpuSse42())) {
      enabled_paths = enabled_paths & ~Path::kSse42;
      // Sanity check.
      RUY_DCHECK((enabled_paths & Path::kSse42) == Path::kNone);
    }
  }

  if ((enabled_paths & Path::kAvx2) != Path::kNone) {
    if (!(HaveBuiltPathForAvx2() && DetectCpuAvx2())) {
      enabled_paths = enabled_paths & ~Path::kAvx2;
      // Sanity check.
      RUY_DCHECK((enabled_paths & Path::kAvx2) == Path::kNone);
    }
  }

  if ((enabled_paths & Path::kAvx512) != Path::kNone) {
    if (!(HaveBuiltPathForAvx512() && DetectCpuAvx512())) {
      enabled_paths = enabled_paths & ~Path::kAvx512;
      // Sanity check.
      RUY_DCHECK((enabled_paths & Path::kAvx512) == Path::kNone);
    }
  }

  // TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete /
  // placeholder. Optimization is not finished. In particular the dimensions of
  // the kernel blocks can be changed as desired.
  //
  if ((enabled_paths & Path::kAvxVnni) != Path::kNone) {
    if (!(HaveBuiltPathForAvxVnni() && DetectCpuAvxVnni())) {
      enabled_paths = enabled_paths & ~Path::kAvxVnni;
      // Sanity check.
      RUY_DCHECK((enabled_paths & Path::kAvxVnni) == Path::kNone);
    }
  }
#endif  // RUY_PLATFORM(X86)

  // Sanity check. We can't possibly have disabled all paths, as some paths
  // are universally available (kReference, kStandardCpp).
  RUY_DCHECK_NE(enabled_paths, Path::kNone);
  return enabled_paths;
}

const std::vector<std::unique_ptr<PerThreadState>>&
ContextInternal::GetPerThreadStates(Context* context, int thread_count) {
  while (context->per_thread_states_.size() <
         static_cast<std::size_t>(thread_count)) {
    context->per_thread_states_.emplace_back(new PerThreadState);
  }
  return context->per_thread_states_;
}

Allocator* ContextInternal::GetMainAllocator(Context* context) {
  if (!context->main_allocator_) {
    context->main_allocator_.reset(new Allocator);
  }
  return context->main_allocator_.get();
}

PrepackedCache* ContextInternal::GetPrepackedCache(Context* context) {
  if (!context->prepacked_cache_) {
    context->prepacked_cache_.reset(new PrepackedCache);
  }
  return context->prepacked_cache_.get();
}

Tuning ContextInternal::GetMainThreadTuning(Context* context) {
  const auto& per_thread_states = GetPerThreadStates(context, 1);
  TuningResolver* tuning_resolver = &per_thread_states[0]->tuning_resolver;
  tuning_resolver->SetTuning(context->explicit_tuning_);
  return tuning_resolver->Resolve();
}

}  // namespace ruy
