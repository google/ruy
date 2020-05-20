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

#include "ruy/path.h"

#include "ruy/check_macros.h"
#include "ruy/detect_arm.h"
#include "ruy/detect_x86.h"
#include "ruy/have_built_path_for.h"

namespace ruy {

Path DetectRuntimeSupportedPaths(Path paths_to_test) {
  Path result = kNonArchPaths;

#if RUY_PLATFORM(ARM)
  if ((paths_to_test & Path::kNeonDotprod) != Path::kNone) {
    if (DetectDotprod()) {
      result = result | Path::kNeonDotprod;
    }
  }
#endif  // RUY_PLATFORM(ARM)

#if RUY_PLATFORM(X86)
  // TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete /
  // placeholder. Optimization is not finished. In particular the dimensions of
  // the kernel blocks can be changed as desired.
  //
  if ((paths_to_test & Path::kSse42) != Path::kNone) {
    if (HaveBuiltPathForSse42() && DetectCpuSse42()) {
      result = result | Path::kSse42;
    }
  }

  if ((paths_to_test & Path::kAvx2) != Path::kNone) {
    if (HaveBuiltPathForAvx2() && DetectCpuAvx2()) {
      result = result | Path::kAvx2;
    }
  }

  if ((paths_to_test & Path::kAvx512) != Path::kNone) {
    if (HaveBuiltPathForAvx512() && DetectCpuAvx512()) {
      result = result | Path::kAvx512;
    }
  }

  // TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete /
  // placeholder. Optimization is not finished. In particular the dimensions of
  // the kernel blocks can be changed as desired.
  //
  if ((paths_to_test & Path::kAvxVnni) != Path::kNone) {
    if (HaveBuiltPathForAvxVnni() && DetectCpuAvxVnni()) {
      result = result | Path::kAvxVnni;
    }
  }
#endif  // RUY_PLATFORM(X86)

  // Sanity check. The any bit set in return value must be set either in
  // paths_to_test or in kNonArchPaths.
  RUY_DCHECK_EQ(result & ~(paths_to_test | kNonArchPaths), Path::kNone);
  return result;
}

}  // namespace ruy
