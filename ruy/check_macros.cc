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

#include "ruy/check_macros.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace ruy {
namespace check_macros {

void FatalError(const char *format, ...) {
  va_list args;
  va_start(args, format);

  {
    va_list args_for_vfprintf;
    va_copy(args_for_vfprintf, args);
    vfprintf(stderr, format, args_for_vfprintf);
    va_end(args_for_vfprintf);
  }

#ifdef __ANDROID__
  {
    va_list args_for_android_log_vprint;
    va_copy(args_for_android_log_vprint, args);
    __android_log_vprint(ANDROID_LOG_FATAL, "ruy", format,
                         args_for_android_log_vprint);
    va_end(args_for_android_log_vprint);
  }
#endif

  va_end(args);

  abort();
}

}  // end namespace check_macros
}  // end namespace ruy
