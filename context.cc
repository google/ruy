#include "context.h"

#include "check_macros.h"
#include "detect_dotprod.h"

namespace ruy {

void Context::SetRuntimeEnabledPaths(Path paths) {
  runtime_enabled_paths_ = paths;
}

Path Context::GetRuntimeEnabledPaths() {
  // This function should always return the same value on a given machine.
  // When runtime_enabled_paths_ has its initial value kNone, it performs
  // some platform detection to resolve it to specific Path values.

  // Fast path: already resolved.
  if (runtime_enabled_paths_ != Path::kNone) {
    return runtime_enabled_paths_;
  }

  // Need to resolve now. Start by considering all paths enabled.
  runtime_enabled_paths_ = kAllPaths;

  // Now selectively disable paths that aren't supported on this machine.
  if ((runtime_enabled_paths_ & Path::kNeonDotprodAsm) != Path::kNone) {
    if (!DetectDotprod()) {
      runtime_enabled_paths_ = runtime_enabled_paths_ ^ Path::kNeonDotprodAsm;
      // Sanity check.
      RUY_DCHECK((runtime_enabled_paths_ & Path::kNeonDotprodAsm) ==
                 Path::kNone);
    }
  }

  // Sanity check. We can't possibly have disabled all paths, as some paths
  // are universally available (kReference, kStandardCpp).
  RUY_DCHECK(runtime_enabled_paths_ != Path::kNone);
  return runtime_enabled_paths_;
}

}  // namespace ruy
