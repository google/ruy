#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PMU_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PMU_H_

#include <cstdint>

namespace ruy {

class PmuEventsPrivate;

class PmuEvents {
 public:
  PmuEvents();
  ~PmuEvents();
  void StartRecording();
  void StopRecording();
  float L1AccessCount() const;
  float L1RefillCount() const;
  float L2RefillCount() const;
  float L3RefillCount() const;
  float BranchMispredictionRate() const;
  float FrontendStallRate() const;
  float BackendStallRate() const;

 private:
  PmuEventsPrivate* priv = nullptr;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PMU_H_
