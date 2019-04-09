// This file is a fork of gemmlowp's allocator.h, under Apache 2.0 license.

#include "allocator.h"

#include "check_macros.h"

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace ruy {

void Allocator::Commit() {
  RUY_DCHECK(!committed_);

  if (reserved_bytes_ > storage_size_) {
    DeallocateStorage();
    storage_size_ = round_up_pot(reserved_bytes_);
    storage_ = aligned_alloc(kAlignment, storage_size_);
  }

  RUY_DCHECK(!storage_size_ || storage_);
  committed_ = true;
}

void Allocator::Decommit() {
  RUY_DCHECK(committed_);
  committed_ = false;
  generation_++;

  reserved_blocks_ = 0;
  reserved_bytes_ = 0;
}

void Allocator::DeallocateStorage() {
  RUY_DCHECK(!committed_);
  aligned_free(storage_);
  storage_size_ = 0;
}

#ifdef _WIN32
void* Allocator::aligned_alloc(size_t alignment, size_t size) {
  return _aligned_malloc(size, alignment);
}

void Allocator::aligned_free(void* memptr) { _aligned_free(memptr); }
#else
void* Allocator::aligned_alloc(size_t alignment, size_t size) {
  void* memptr;
  if (posix_memalign(&memptr, alignment, size)) {
    memptr = nullptr;
  }
  return memptr;
}

void Allocator::aligned_free(void* memptr) { free(memptr); }
#endif

}  // namespace ruy
