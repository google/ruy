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

#include "ruy/create_trmul_params.h"

#include "ruy/mat.h"
#include "ruy/opt_set.h"

namespace ruy {
namespace detail {

void CreatePackedLayout(const MatLayout& src, const Type& scalar,
                        const KernelLayout& kernel_layout, PMatLayout* packed) {
  packed->order = Order::kColMajor;
  packed->rows = round_up_pot(src.rows, kernel_layout.rows);
  packed->cols = round_up_pot(src.cols, kernel_layout.cols);
  packed->kernel = kernel_layout;
  int inner_size = packed->rows;
  if (RUY_OPT(AVOID_ALIASING)) {
    packed->stride =
        (inner_size * scalar.size) % 1024 ? inner_size : inner_size + 64;
  } else {
    packed->stride = inner_size;
  }
}

}  // namespace detail
}  // namespace ruy
