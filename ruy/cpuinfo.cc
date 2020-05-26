#include "ruy/cpuinfo.h"

#include "ruy/platform.h"

#define RUY_HAVE_CPUINFO (!RUY_PPC)

#if RUY_HAVE_CPUINFO

#include <cpuinfo.h>

namespace ruy {

CpuInfo::~CpuInfo() {
  if (init_status_ == InitStatus::kInitialized) {
    cpuinfo_deinitialize();
  }
}

bool CpuInfo::EnsureInitialized() {
  if (init_status_ == InitStatus::kNotYetAttempted) {
    init_status_ =
        cpuinfo_initialize() ? InitStatus::kInitialized : InitStatus::kFailed;
  }
  return init_status_ == InitStatus::kInitialized;
}

bool CpuInfo::NeonDotprod() {
  return EnsureInitialized() && cpuinfo_has_arm_neon_dot();
}

bool CpuInfo::Sse42() {
  return EnsureInitialized() && cpuinfo_has_x86_sse4_2();
}

bool CpuInfo::Avx2() { return EnsureInitialized() && cpuinfo_has_x86_avx2(); }

bool CpuInfo::Avx512() {
  return EnsureInitialized() && cpuinfo_has_x86_avx512f() &&
         cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512cd() &&
         cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512vl();
}

bool CpuInfo::AvxVnni() {
  return EnsureInitialized() && cpuinfo_has_x86_avx512vnni();
}

}  // namespace ruy

#else  // not RUY_HAVE_CPUINFO

namespace ruy {
CpuInfo::~CpuInfo() {}
}  // namespace ruy

#endif
