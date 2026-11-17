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

// Library doing minimal CPU detection to decide what to tune asm code for.
//
// # Tuning vs Path
//
// Tunings are merely local variations of optimized code paths, that are
// drop-in replacements for each other --- the input and output data layouts
// are identical.  By contrast, what ruy calls a Path dictates its own
// data layouts. For example, Path::kNeonDotprod will use different
// layouts compared to Path::kNeon; but within each, different tunings
// will share that same layout.
//
// # Tuning for in-order vs out-of-order CPUs
//
// Out-of-order CPUs are by definition relatively insensitive to small-scale
// asm details (which is what "tuning" is about), while in-order CPUs such as
// Cortex-A53 and Cortex-A55 require careful instruction scheduling and
// core-specific prefetch tuning:
//  * For Cortex-A53 (kA53), 64-bit loads + INS are used and software prefetches
//    (prfm) in the inner loop are beneficial.
//  * For Cortex-A55 (kA55, covering both A55r0 pre-dotprod and A55r1+ with
//    dotprod), 64-bit loads + INS are used and software prefetches (prfm) in
//    the inner loop are omitted since they cost a full cycle.
#ifndef RUY_RUY_TUNE_H_
#define RUY_RUY_TUNE_H_

#include "ruy/cpuinfo.h"
#include "ruy/opt_set.h"
#include "ruy/platform.h"
#include "ruy/time.h"

namespace ruy {

enum class Tuning {
  // kAuto means please use auto-detection. It's the default in the
  // user-visible parts (see Context). It's meant to be resolved to an
  // actual tuning at some point by means of TuningResolver.
  kAuto,
  // Use code not tuned for any particular CPU, typically performing well
  // on out-of-order cores that don't require as much tuning.
  kGeneric,
  // Use code tuned for Cortex-A53 CPUs: in-order cores that require using
  // non-NEON ARM 64-bit loads together with INS instructions to dual-issue
  // 64-bit loads, and where software prefetches (prfm) in the inner loop are a
  // performance win.
  kA53,
  // Use code tuned for Cortex-A55 CPUs (A55r0 pre-dotprod and A55r1+ with
  // dotprod): in-order cores that benefit from 64-bit loads, but where
  // software prefetches (prfm) in the inner loop cost a full cycle and are
  // omitted.
  kA55,
  // Use code tuned for Cortex-X1 CPUs. Currently, the driver to distinguish
  // this CPU is the get maximum performance on the dotprod kernels, where we
  // attain high performance simply by avoiding any manual loop unrolling. As a
  // purely performance oriented microarchitecture, there will likely be
  // additional reasons to distinguish the X1 from other CPUs.
  kX1
};

// Why a TuningResolver class?
//
// Ideally, this Library would offer a single function,
//   Tuning GetCurrentCPUTuning();
//
// However, determining information about the current CPU is not necessarily
// cheap, so we currently cache that and only invalidate/reevaluate after
// a fixed amount of time. This need to store state is why this library
// has to expose a class, TuningResolver, not just a function.
class TuningResolver {
 public:
  TuningResolver();

  // Allows the user to specify an explicit Tuning value, bypassing auto
  // detection; or to specify Tuning::kAuto, reverting to auto detection.
  void SetTuning(Tuning tuning) { unresolved_tuning_ = tuning; }

  // Get an actual tuning --- that is the function that this class wanted to be.
  Tuning Resolve(CpuInfo* cpuinfo);

 private:
  TuningResolver(const TuningResolver&) = delete;

  // Perform the tuning resolution now. That may typically use EvalRatio and
  // ThresholdRatio, but an implementation may use a different approach instead.
  Tuning ResolveNow(CpuInfo* cpuinfo);

  // The tuning as specified by the user, before actual resolution happens
  // i.e. before querying any specifics of the current CPU.
  // The default value kAuto means try to auto-detect. Other values mean
  // bypass auto-detect, use explicit value instead. See SetTuning().
  Tuning unresolved_tuning_ = Tuning::kAuto;
  // Cached last resolved tuning.
  Tuning last_resolved_tuning_ = Tuning::kAuto;
  // Timepoint of cached last resolved tuning, for invalidation purposes.
  TimePoint last_resolved_timepoint_;
  // Cached last resolved tunings that are older than this age are invalid.
  const Duration expiry_duration_;
};

}  // namespace ruy

#endif  // RUY_RUY_TUNE_H_
