/* Copyright 2020 Google LLC. All Rights Reserved.

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

#include "ruy/have_built_path_for.h"
#include "ruy/opt_set.h"

namespace ruy {

#if RUY_PLATFORM_X86
// IMPORTANT:
// These patterns must match those in the pack and kernel cc files.
#if !(RUY_PLATFORM_AVX && RUY_OPT(ASM))

bool HaveBuiltPathForAvx() { return false; }

#else  // RUY_PLATFORM_AVX && RUY_OPT(ASM)

bool HaveBuiltPathForAvx() { return true; }

#endif  // RUY_PLATFORM_AVX && RUY_OPT(ASM)
#endif  // RUY_PLATFORM_X86

}  // namespace ruy
