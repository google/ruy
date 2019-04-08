// This file is a fork of gemmlowp's allocator.h, under Apache 2.0 license.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_ALLOCATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_ALLOCATOR_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "check_macros.h"
#include "size_util.h"

namespace ruy {

// A simple allocator allowing:
//   - Coalescing multiple allocations into a single heap-allocated buffer.
//   - Keeping that heap buffer and growing it as needed for future usage.
// The idea is to minimize the number of heap allocations and to quickly
// converge to a steady state where no heap allocations are performed anymore
// as the buffer is large enough and the need to grow it occurs more and more
// rarely.
//
// In order to provide this functionality with very little code and overhead,
// this allocator supports only a very rigid usage pattern. There are two
// states: "uncommitted" and "committed". Uncommitted is the initial state.
// In that state, all one can do is to call Reserve to let the allocator
// know about the blocks that one wants to allocate, and get Handles for them.
// This Reserve logic is local to the allocator and no heap allocation occurs
// at this point.  Then one calls Commit, transitioning to committed state,
// which is where allocator's internal heap buffer is grown as necessary.
// In that state, all one can do is to call GetPointer to query the actual
// pointers to the allocated blocks within that buffer. Finally, one calls
// Decommit to revert to the initial uncommitted state. The allocator's
// internal buffer is not actually deallocated until the destruction of the
// Allocator.
//
// We call 'generation' the time span between consecutive Decommit calls.
// Handles created by Reserve at one generation are only valid within that
// generation, i.e. become invalid at the next Decommit.
//
// Example usage:
//
// struct Foo { float x; int y; };
//
// void DoSomeWork(Allocator* allocator) {
//   Allocator::Handle<Foo> handle_foo
//   Allocator::Handle<int> handle_bar;
//   // reserve a buffer of 1 Foo
//   Reserve(allocator, 1, &handle_foo);
//   // reserve a buffer of 10 ints
//   Reserve(allocator, 10, &handle_bar);
//   allocator->Commit();
//   Foo* ptr_foo = allocator->GetPointer(handle_foo);
//   int* ptr_bar = allocator->GetPointer(handle_bar);
//   ... do some work ...
//   allocator->Decommit();
// }
//
class Allocator {
 public:
  // See generation_.
  //
  // Intentionally narrow. Since this generation count is only used for
  // debugging purposes (to catch bad usage patterns mixing handles from
  // different generations) it's not terrible to actually have this wrap around
  // at the end of this type's range, and then we'd rather have this wrap-around
  // happen quickly enough that it's actually testable.
  using generation_t = std::uint8_t;
  static_assert(!std::is_signed<generation_t>::value,
                "generation_t must be unsigned to have well-defined "
                "wrap-around overflow behavior");

  // We haven't needed more than this many blocks so far,
  // and since the usage pattern is fixed, there is no point in allowing more
  // until we need to.
  using index_t = std::int8_t;
  static constexpr index_t kMaxBlocks = 16;

  // Alignment of allocated blocks.
  //
  // Considerations:
  //  - This needs to be at least the alignment of any usual data type.
  //  - It's useful that this is at least the size of a cache line to limit
  //    possible cache side effects (if only on performance behavior).
  //  - It's useful that this is at least the size of SIMD registers, as
  //    some SIMD instruction sets have at least performance behavior
  //    differences (e.g. NEON) or even different requirements (e.g. SSE)
  //    based on that.
  //  - It's useful that this is at least the size of an "exclusive reservation
  //    granule" on ARM, meaning that if we use this Allocator to allocate
  //    an atomic variable, there will be no side effects from other things
  //    contending for exclusive/atomic memory accesses to it. While the
  //    ARM reference manual mentions that this granule size may be as large
  //    as 2048 bytes, in practice we observe it to be 64 bytes. It can
  //    be queried cheaply, at runtime, from userspace, if needed.
  static constexpr std::size_t kAlignment = 64;

  // A handle on a reserved block. The user obtains
  // one by calling Reserve() and, after committing,
  // passes it to GetPointer().
  //
  // These handles are just direct indices (see index_ here) into the
  // allocator's internal table of reserved blocks offsets
  // (reserved_blocks_offsets_). So these handles are only valid for a
  // given generation of the allocator: there must be equality between
  // handle.generation_ and allocator.generation_ for the handle to be
  // valid to use with the allocator.
  //
  // The templatization on T is how we ensure type safety. Because handles
  // are only user-facing objects that the user passes to Reserve and GetPointer
  // we can ensure type safety in these functions while keeping the Allocator
  // implementation un-templatized.
  template <typename T>
  class Handle {
   public:
    Handle() {
      // It's useful that Handle has a small size, as we can have a number of
      // these on the stack.
      static_assert(sizeof(Handle) <= sizeof(void*), "");
      // Enforce that kAlignmnet meets the requirements of type T
      static_assert(kAlignment % alignof(T) == 0, "");
    }

   private:
    // The index into Allocator::reserved_blocks_offsets_.
    // Initialized with a bogus value to catch erroneous usage patterns
    // (typically, GetPointer on a newly-constructed Handle not returned
    // by Reserve).
    index_t index_ = kMaxBlocks;
    // The generation corresponding to Allocator::generation_.
    // We don't bother initializing with a bogus value because due to
    // wrapping around behavior at the end of generation_t, there is no
    // really bogus value here.
    generation_t generation_ = 0;
    friend class Allocator;
  };

  Allocator() {}

  ~Allocator() {
    RUY_DCHECK(!committed_);
    RUY_DCHECK(!reserved_blocks_);
    DeallocateStorage();
  }

  void Commit();
  void Decommit();

  // Reserves a block sized for n elements of type T, and
  // returns a handle to it. Must be called before committing.
  template <typename T>
  void Reserve(std::size_t n, Handle<T>* h) {
    RUY_DCHECK(!committed_);
    RUY_DCHECK_LT(reserved_blocks_, kMaxBlocks);
    const std::size_t bytes = round_up_pot(n * sizeof(T), kAlignment);
    const std::size_t offset = reserved_bytes_;
    const int index = reserved_blocks_;

    reserved_blocks_offsets_[index] = offset;
    h->index_ = index;
    h->generation_ = generation_;

    reserved_blocks_++;
    reserved_bytes_ += bytes;
  }

  // Returns the pointer to the allocated buffer for the given handle.
  // Must be called after committing.
  template <typename T>
  T* GetPointer(const Handle<T>& h) const {
    RUY_DCHECK(committed_);
    RUY_DCHECK_LT(h.index_, reserved_blocks_);
    RUY_DCHECK_EQ(h.generation_, generation_);
    std::size_t offset = reserved_blocks_offsets_[h.index_];
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(storage_) + offset;
    return reinterpret_cast<T*>(addr);
  }

 private:
  Allocator(const Allocator&) = delete;

  void* aligned_alloc(size_t alignment, size_t size);
  void aligned_free(void* memptr);

  void DeallocateStorage();

  // Set to true by Commit() and to false by Decommit().
  bool committed_ = false;

  // The actually allocated storage size and buffer pointer.
  std::size_t storage_size_ = 0;
  void* storage_ = nullptr;

  // The number of blocks that have been reserved by Reserve().
  int reserved_blocks_ = 0;
  // The number of bytes that have been reserved by Reserve().
  std::size_t reserved_bytes_ = 0;
  // The offsets of reserved blocks into the storage buffer.
  std::size_t reserved_blocks_offsets_[kMaxBlocks];

  // The 'generation' is incremented on Decommit() and allows catching
  // bad GetPointer() calls still referring to a previous commit.
  generation_t generation_ = 0;
};

// Convenience alternative function for the Reserve method, avoiding the
// ugly C++ syntax for calling template methods on templated object, and
// allowing for template parameter deduction thanks to taking the handle
// as an out-argument instead of returning it by value.
template <typename T>
void Reserve(Allocator* allocator, std::size_t n, Allocator::Handle<T>* h) {
  allocator->Reserve(n, h);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_ALLOCATOR_H_
