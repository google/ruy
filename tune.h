#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TUNE_H_

#include <cstdint>

namespace ruy {

enum class Tuning { kAuto, kOutOfOrder, kInOrder };

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
