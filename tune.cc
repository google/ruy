#include "tune.h"

#include <algorithm>
#include <cstdint>

#include "opt_set.h"
#include "time.h"

namespace ruy {

namespace {

bool IsTimestampRecent(std::uint64_t old_timestamp, std::uint64_t new_timestamp,
                       std::uint64_t expiry) {
  return (new_timestamp - old_timestamp) < expiry;
}

std::uint64_t GetTimestampExpiry() {
  static constexpr int kExpiryInverseSecs = 4;
  return TimeFrequency() / kExpiryInverseSecs;
}

}  // namespace

#ifdef __aarch64__

namespace {

void PoorlyOrderedKernel(int iters) {
  asm volatile(
      "mov w0, %w[iters]\n"
      "1:\n"
      "subs w0, w0, #1\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "bne 1b\n" ::[iters] "r"(iters)
      : "cc", "x0", "v0", "v1", "v2", "v3");
}

void NicelyOrderedKernel(int iters) {
  asm volatile(
      "mov w0, %w[iters]\n"
      "1:\n"
      "subs w0, w0, #1\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "mul v0.4s, v0.4s, v0.4s\n"
      "mul v1.4s, v1.4s, v1.4s\n"
      "mul v2.4s, v2.4s, v2.4s\n"
      "mul v3.4s, v3.4s, v3.4s\n"
      "bne 1b\n" ::[iters] "r"(iters)
      : "cc", "x0", "v0", "v1", "v2", "v3");
}

}  // namespace

float TuningResolver::EvalRatio() {
  // With the current settings, 400 iterations and 4 repeats, this test has
  // a latency of roughly 80 microseconds on a Cortex-A53 at 1.4 GHz.
  static constexpr int kLoopIters = 400;
  static constexpr int kRepeats = 4;

  std::uint64_t timing_poorly_ordered =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t timing_nicely_ordered =
      std::numeric_limits<std::uint64_t>::max();

  for (int r = 0; r < kRepeats; r++) {
    std::uint64_t t0 = TimeNowBarrier();
    PoorlyOrderedKernel(kLoopIters);
    std::uint64_t t1 = TimeNowBarrier();
    NicelyOrderedKernel(kLoopIters);
    std::uint64_t t2 = TimeNowBarrier();
    timing_poorly_ordered = std::min(timing_poorly_ordered, t1 - t0);
    timing_nicely_ordered = std::min(timing_nicely_ordered, t2 - t1);
  }

  return static_cast<float>(timing_nicely_ordered) /
         static_cast<float>(timing_poorly_ordered);
}

float TuningResolver::ThresholdRatio() {
  // Empirically (see :tune_tool) determined threshold to distinguish in-order
  // Cortex-A53/A55 cores from out-of-order Cortex-A57/A73/A75/A76 cores. Based
  // on these experimental results, which were obtained with much lower
  // (kLoopIters=1000, kRepeats=1) so as to make them resilient to noise, we
  // have:
  //
  // CPU core type | in/out of order | observed ratio
  //               |                 | (timing_nicely_ordered /
  //               timing_poorly_ordered)
  // --------------+-----------------+-----------------------------------------
  // Cortex-A53    | in-order        | 0.32 -- 0.329
  // Cortex-A55    | in-order        | 0.319 -- 0.325
  // Cortex-A55r1  | in-order        | 0.319 -- 0.325
  // Cortex-A57    | out-of-order    | 0.99 -- 1.01
  // Cortex-A73    | out-of-order    | 0.922 -- 0.927
  // Cortex-A75    | out-of-order    | 0.921 -- 0.93
  // Cortex-A76    | out-of-order    | 1
  // Kryo (pixel1) | out-of-order    | 0.73 -- 0.76
  //
  // Thus the allowable range for the threshold is [0.35 .. 0.70].
  // We pick a value closer to the upper bound because really any out-of-order
  // CPU should by definition produce a ratio close to 1.
  return 0.65f;
}

Tuning TuningResolver::ResolveNow() {
  const bool is_probably_inorder = EvalRatio() < ThresholdRatio();
  return is_probably_inorder ? Tuning::kInOrder : Tuning::kOutOfOrder;
}

#else  // not defined __aarch64__

float TuningResolver::EvalRatio() { return 0; }
float TuningResolver::ThresholdRatio() { return 0; }

Tuning TuningResolver::ResolveNow() { return Tuning::kOutOfOrder; }

#endif

Tuning TuningResolver::Resolve() {
#if (defined RUY_OPT_SET) && !(RUY_OPT_SET & RUY_OPT_TUNING)
  return Tuning::kOutOfOrder;
#endif
  if (unresolved_tuning_ != Tuning::kAuto) {
    return unresolved_tuning_;
  }
  std::uint64_t new_timestamp = TimeNowRelaxed();
  if (!timestamp_expiry_) {
    timestamp_expiry_ = GetTimestampExpiry();
  }
  if (last_resolved_tuning_ != Tuning::kAuto &&
      IsTimestampRecent(last_resolved_timestamp_, new_timestamp,
                        timestamp_expiry_)) {
    return last_resolved_tuning_;
  }
  last_resolved_timestamp_ = new_timestamp;
  last_resolved_tuning_ = ResolveNow();
  return last_resolved_tuning_;
}

}  // namespace ruy
