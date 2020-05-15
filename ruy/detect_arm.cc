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

/* Detection of dotprod instructions on ARM.
 * The current Linux-specific code relies on the feature, available in
 * Linux 4.11 and newer, whereby the kernel allows userspace code to perform
 * some `mrs` instructions, reading certain feature registers normally available
 * only to privileged code. It is described here,
 * https://www.kernel.org/doc/html/latest/arm64/cpu-feature-registers.html
 *
 * Starting with Linux 4.15 (backported to 4.14.151), the kernel also directly
 * exposes the bit that we need to userspace by means of HWCAP_ASIMDDP.
 * Unfortunately, we need this on earlier 4.14.x kernels.
 */

#include "ruy/detect_arm.h"

#include <cstdint>

#if defined __linux__ && defined __aarch64__
#include <sys/auxv.h>
#endif

namespace ruy {

namespace {

#if defined __linux__ && defined __aarch64__
bool IsCpuidAvailable() {
  // This is the value of HWCAP_CPUID in Linux >= 4.11, however we need to
  // support building against older headers for the time being.
  const int kLocalHwcapCpuid = 1 << 11;
  return getauxval(AT_HWCAP) & kLocalHwcapCpuid;
}

bool DetectDotprodByCpuid() {
  // The bit that we need is 'DP', which is bit 47 in ID_AA64ISAR0_EL1:
  // https://developer.arm.com/docs/ddi0595/e/aarch64-system-registers/id_aa64isar0_el1#DP_47
  std::uint64_t x;
  asm("mrs %[x], ID_AA64ISAR0_EL1" : [x] "=r"(x) : :);
  return x & (1ull << 47);
}
#endif

}  // namespace

bool DetectDotprod() {
#if defined __linux__ && defined __aarch64__
  if (IsCpuidAvailable()) {
    return DetectDotprodByCpuid();
  }
#endif

  return false;
}

}  // namespace ruy
