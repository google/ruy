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

#include "ruy/allocator.h"

#include "ruy/system_aligned_alloc.h"

namespace ruy {

Allocator::~Allocator() {
  FreeAll();
  detail::SystemAlignedFree(ptr_);
}

void* Allocator::AllocateSlow(std::ptrdiff_t num_bytes) {
  void* p = detail::SystemAlignedAlloc(num_bytes);
  fallback_blocks_total_size_ += num_bytes;
  fallback_blocks_.push_back(p);
  return p;
}

void Allocator::FreeAll() {
  current_ = 0;
  if (fallback_blocks_.empty()) {
    return;
  }

  // No rounding-up of the size means linear instead of logarithmic
  // bound on the number of allocation in some worst-case calling patterns.
  // This is considered worth it because minimizing memory usage is important
  // and actual calling patterns in applications that we care about still
  // reach the no-further-allocations steady state in a small finite number
  // of iterations.
  std::ptrdiff_t new_size = size_ + fallback_blocks_total_size_;
  detail::SystemAlignedFree(ptr_);
  ptr_ = detail::SystemAlignedAlloc(new_size);
  size_ = new_size;

  for (void* p : fallback_blocks_) {
    detail::SystemAlignedFree(p);
  }
  fallback_blocks_.clear();
  fallback_blocks_total_size_ = 0;
}

}  // namespace ruy
