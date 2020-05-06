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

#ifndef RUY_RUY_ALLOCATOR_H_
#define RUY_RUY_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ruy/size_util.h"
#include "ruy/system_aligned_alloc.h"

namespace ruy {

// Specialized allocator designed to converge to a steady-state where all
// allocations are bump-ptr allocations from an already-allocated buffer.
//
// To support these constraints, this allocator only supports two
// operations.
// - AllocateBytes/Allocate<Pointer>: allocates a pointer to storage of a
// specified size, which will be aligned to kMinimumBlockAlignment.
// - FreeAll: frees all previous allocations (but retains the internal
// buffer to minimize future calls into the system allocator).
//
// This class is specialized for supporting just those two operations
// under this specific steady-state usage pattern. Extending this class
// with new allocation interfaces that don't fit that pattern is probably not
// the right choice. Instead, build a new class on top of
// SystemAlignedAlloc/SystemAlignedFree.
//
// All operations happen on aligned blocks for simplicity.
//
// Theory of operation:
//
// - ptr_, current_, and size_ implement a basic bump-ptr allocator.
//
// - in AllocateBytes, the fast path is just a bump-ptr
// allocation. If our bump-ptr allocator doesn't have enough space for an
// allocation, then we allocate a block from the system allocator to
// service the allocation request. We save that block in fallback_blocks_
// and track the total size of the fallback blocks in
// fallback_blocks_total_size_.
//
// - in FreeAll, the fast path just resets the bump-ptr allocator. If
// there are any fallback blocks, we free them and reallocate the
// bump-ptr allocator's buffer so that the next sequence of allocations
// will hopefully not need any fallback blocks.
class Allocator final {
 public:
  ~Allocator();

  void* AllocateBytes(std::ptrdiff_t num_bytes) {
    if (num_bytes == 0) {
      return nullptr;
    }
    const std::ptrdiff_t rounded_num_bytes =
        round_up_pot(num_bytes, detail::kMinimumBlockAlignment);
    if (void* p = AllocateFast(rounded_num_bytes)) {
      return p;
    }
    return AllocateSlow(rounded_num_bytes);
  }

  template <typename Pointer>
  void Allocate(std::ptrdiff_t count, Pointer* out) {
    using T = typename std::pointer_traits<Pointer>::element_type;
    *out = static_cast<T*>(AllocateBytes(count * sizeof(T)));
  }

  void FreeAll();

 private:
  void operator=(const Allocator&) = delete;
  void* AllocateSlow(std::ptrdiff_t num_bytes);

  void* AllocateFast(std::ptrdiff_t num_bytes) {
    if (current_ + num_bytes > size_) {
      return nullptr;
    }
    void* ret = static_cast<char*>(ptr_) + current_;
    current_ += num_bytes;
    return ret;
  }

  void* ptr_ = nullptr;
  std::ptrdiff_t current_ = 0;
  std::ptrdiff_t size_ = 0;
  std::vector<void*> fallback_blocks_;
  std::ptrdiff_t fallback_blocks_total_size_ = 0;
};

}  // namespace ruy

#endif  // RUY_RUY_ALLOCATOR_H_
