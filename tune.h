// Library doing minimal CPU detection to decide what to tune asm code for.
//
// # Tuning vs Path
//
// Tunings are merely local variations of optimized code paths, that are
// drop-in replacements for each other --- the input and output data layouts
// are identical.  By contrast, what ruy calls a Path dictates its own
// data layouts. For example, Path::kNeonDotprodAsm will use different
// layouts compared to Path::kNeonAsm; but within each, different tunings
// will share that same layout.
//
// # Tuning is for now only based on 1 bit: OutOfOrder / InOrder
//
// In practice, each of our asm code paths only needs one bit information to
// decide on tuning: whether the CPU is out-of-order or in-order.
// That is because out-of-order CPUs are by definition relatively insensitive
// to small-scale asm details (which is what "tuning" is about); and for each
// asm code path, there tends to be one main in-order CPU architecture that
// we focus our tuning effort on. Examples:
//  * For Path::kNeonAsm, the main in-order CPU is Cortex-A53/A55 (pre-dotprod)
//  * For Path::kNeonDotprodAsm, the main in-order CPU is Cortex-A55r1 (dotprod)
//
// Because having tuned code paths is a compromise of efficiency gains
// versus implementation effort and code size, we are happy to stop at just this
// single bit of information, OutOfOrder/InOrder, at least in the current CPU
// landscape. This could change in the future.
//
// # Implementation notes and alternatives.
//
// The current implementation uses a nano-benchmark, see tune.cc.
// That is why it's quite expensive, making caching /
// statefulness necessary (see TuningResolver class comment).
//
// An interesting alternative, which was explained to us by Marat Dukhan
// (maratek@) after this was implemented, would be to use the
// getcpu(2) system call on Linux. This returns a
// numeric CPU identifier that could be mapped to a OutOfOrder/InOrder
// classification given additional information about the CPU.  Such
// additional information could be obtained by the cpuinfo library,
//   https://github.com/pytorch/cpuinfo
// which obtains this information mainly from parsing /proc/cpuinfo.
// Pros:
//   * Would remove the need for the relatively expensive nano-benchmark
//     (dozens of microseconds, which have to be reevaluated again several
//     times per second).
//   * Would conceivably be more reliable.
// Cons:
//   * Linux-specific.
//   * Modest binary size increase (Marat mentioned the cpuinfo lib is 20k).
//   * Won't support exactly 100% of devices (nonstandard /proc/cpuinfo etc).
//
// We could also have both:
//  * Maybe by trying getcpu first if supported, then falling back to a
//    nano-benchmark.
//  * Maybe using getcpu in conjunction with the nano-benchmark to cache
//    per-CPU-id nano-benchmark results.
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_

#include <cstdint>

namespace ruy {

enum class Tuning {
  // kAuto means please use auto-detection. It's the default in the
  // user-visible parts (see Context). It's meant to be resolved to an
  // actual tuning at some point by means of TuningResolver.
  kAuto,
  // Target an out-order CPU. Example: ARM Cortex-A75.
  kOutOfOrder,
  // Target an in-order CPU. Example: ARM Cortex-A55.
  kInOrder
};

// Why a TuningResolver class?
//
// Ideally, this Library would offer a single function,
//   Tuning GetCurrentCPUTuning();
//
// However, determining information about the current CPU is not necessarily,
// cheap, so we currently cache that and only invalidate/reevaluate after
// a fixed amount of time. This need to store state is why this library
// has to expose a class, TuningResolver, not just a function.
class TuningResolver {
 public:
  TuningResolver() {}

  void SetExplicitTuning(Tuning explicit_tuning) {
    unresolved_tuning_ = explicit_tuning;
  }

  Tuning Resolve();

 private:
  TuningResolver(const TuningResolver&) = delete;

  Tuning unresolved_tuning_ = Tuning::kAuto;
  Tuning last_resolved_tuning_ = Tuning::kAuto;
  std::uint64_t last_resolved_timestamp_ = 0;
  std::uint64_t timestamp_expiry_ = 0;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_