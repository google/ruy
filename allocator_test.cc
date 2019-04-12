#include "allocator.h"

#include <cstdlib>

#include "testing/base/public/gunit.h"

namespace ruy {
namespace {

TEST(AllocatorTest, ReturnsValidMemory) {
  Allocator allocator;
  int *p;
  allocator.Allocate(1, &p);
  ASSERT_NE(p, nullptr);

  // If this is bogus memory, ASan will cause this test to fail.
  *p = 42;

  allocator.FreeAll();
}

TEST(AllocatorTest, NoLeak) {
  Allocator allocator;
  // Allocate and free some ridiculously large total amount of memory, so
  // that a leak will hopefully cause some sort of resource exhaustion.
  //
  // Despite the large number of allocations, this test is actually quite
  // fast, since our fast-path allocation logic is very fast.
  constexpr int kNumAllocations = 100 * 1024;
  constexpr int kAllocationSize = 1024 * 1024;
  for (int i = 0; i < kNumAllocations; i++) {
    char *p;
    allocator.Allocate(kAllocationSize, &p);
    allocator.FreeAll();
  }
}

TEST(AllocatorTest, IncreasingSizes) {
  Allocator allocator;
  // Allocate sizes that increase by small amounts across FreeAll calls.
  for (int i = 1; i < 100 * 1024; i++) {
    char *p;
    allocator.Allocate(i, &p);
    allocator.FreeAll();
  }
}

TEST(AllocatorTest, ManySmallAllocations) {
  Allocator allocator;
  // Allocate many small allocations between FreeAll calls.
  for (int i = 0; i < 10 * 1024; i += 100) {
    for (int j = 0; j < i; j++) {
      char *p;
      allocator.Allocate(1, &p);
    }
    allocator.FreeAll();
  }
}

}  // namespace
}  // namespace ruy

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
