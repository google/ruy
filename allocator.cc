#include "allocator.h"

#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace ruy {

namespace detail {

void *AlignedAllocator::SystemAlignedAlloc(std::size_t num_bytes) {
#ifdef _WIN32
  return _aligned_malloc(num_bytes, kAlignment);
#else
  void *ptr;
  if (posix_memalign(&ptr, kAlignment, num_bytes)) {
    return nullptr;
  }
  return ptr;
#endif
}

void AlignedAllocator::SystemAlignedFree(void *ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace detail

}  // namespace ruy
