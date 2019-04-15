// Self-contained tool used to tune the tune code --- see the
// threshold ratios used in tune.cc.

#include <chrono>  // NOLINT(build/c++11)
#include <cstdio>
#include <thread>  // NOLINT(build/c++11)

#include "tune.h"

#ifdef _WIN32
#define getpid() 0
#else
#include <unistd.h>
#endif

namespace ruy {

class TuneTool {
 public:
  static void Query(float* eval, float* threshold) {
    TuningResolver resolver;
    *eval = resolver.EvalRatio();
    *threshold = resolver.ThresholdRatio();
  }
};

}  // namespace ruy

int main() {
  // Infinite loop: the user can hit Ctrl-C
  while (true) {
    float eval;
    float threshold;
    ruy::TuneTool::Query(&eval, &threshold);
    printf("[%d] eval=%.3f %c threshold=%.3f  ==> probably %s...\n", getpid(),
           eval, eval < threshold ? '<' : '>', threshold,
           eval < threshold ? "in-order" : "out-of-order");
    fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
