#include "allocator.h"

namespace ruy {

namespace detail {

void *AlignedAllocator::SystemAlignedAlloc(std::size_t num_bytes) {
  void *ptr;
  if (posix_memalign(&ptr, kAlignment, num_bytes)) {
    return nullptr;
  }
  return ptr;
}

void AlignedAllocator::SystemAlignedFree(void *ptr) { free(ptr); }

}  // namespace detail

}  // namespace ruy
