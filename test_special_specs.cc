// This test covers non-standard specs.

#include "test.h"

namespace ruy {

template <typename AccumScalar, typename DstScalar,
          LoopStructure tLoopStructure>
struct LoopStructureSpec : BasicSpec<AccumScalar, DstScalar> {
  static constexpr LoopStructure kLoopStructure = tLoopStructure;
};

template <typename AccumScalar, typename DstScalar,
          ZeroPointSupport tZeroPointSupport>
struct ZeroPointSupportSpec : BasicSpec<AccumScalar, DstScalar> {
  static constexpr ZeroPointSupport kZeroPointSupport = tZeroPointSupport;
};

template <typename AccumScalar, typename DstScalar>
struct PackedLinearRCCSpec : BasicSpec<AccumScalar, DstScalar> {
  static constexpr LayoutSupport kLayoutSupport =
      LayoutSupport::kPackedLinearRCC;
};

using LhsScalar = RUY_TEST_LHSSCALAR;
using RhsScalar = RUY_TEST_RHSSCALAR;
using AccumScalar = RUY_TEST_ACCUMSCALAR;
using DstScalar = RUY_TEST_DSTSCALAR;

template <LoopStructure tLoopStructure>
void TestLoopStructure() {
  using SpecType = LoopStructureSpec<AccumScalar, DstScalar, tLoopStructure>;
  using TestSetType = TestSet<LhsScalar, RhsScalar, SpecType>;
  for (int size = 1; size < 10; size++) {
    TestLinearAllOrders<TestSetType>(size, size, size);
  }
  TestLinearAllOrders<TestSetType>(3, 5, 78);
  TestLinearAllOrders<TestSetType>(19, 91, 7);
  TestLinearAllOrders<TestSetType>(71, 26, 44);
  TestLinearAllOrders<TestSetType>(81, 93, 72);
}

TEST(TestSpecialSpecs, LoopStructure) {
  static_assert(BasicSpec<std::uint8_t, std::int32_t>::kLoopStructure ==
                    LoopStructure::kAuto,
                "");
  static_assert(BasicSpec<float, float>::kLoopStructure == LoopStructure::kAuto,
                "");
  TestLoopStructure<LoopStructure::kSimple>();
  TestLoopStructure<LoopStructure::kGeneral>();
}

template <ZeroPointSupport tZeroPointSupport>
void TestZeroPointSupport(LhsScalar lhs_zero_point, RhsScalar rhs_zero_point,
                          DstScalar dst_zero_point,
                          ExpectedOutcome expected_outcome) {
  using SpecType =
      ZeroPointSupportSpec<AccumScalar, DstScalar, tZeroPointSupport>;
  using TestSetType = TestSet<LhsScalar, RhsScalar, SpecType>;
  TestSetType test_set;
  test_set.rows = 11;
  test_set.depth = 12;
  test_set.cols = 13;
  test_set.lhs_order = Order::kRowMajor;
  test_set.rhs_order = Order::kColMajor;
  test_set.dst_order = Order::kColMajor;
  test_set.layout_style = LayoutStyle::kPackedLinear;
  test_set.expected_outcome = expected_outcome;
  test_set.lhs_zero_point = lhs_zero_point;
  test_set.rhs_zero_point = rhs_zero_point;
  test_set.dst_zero_point = dst_zero_point;
  test_set.use_specified_zero_points = true;
  test_set.Run();
}

TEST(TestSpecialSpecs, ZeroPointSupport) {
  // Sanity check
  RUY_CHECK_EQ(SymmetricZeroPoint<std::uint8_t>(), 128);
  RUY_CHECK_EQ(SymmetricZeroPoint<std::int8_t>(), 0);

  if (std::is_floating_point<LhsScalar>::value) {
    return;
  }

  TestZeroPointSupport<ZeroPointSupport::kGeneral>(
      SymmetricZeroPoint<LhsScalar>(), SymmetricZeroPoint<RhsScalar>(),
      SymmetricZeroPoint<DstScalar>(), ExpectedOutcome::kSuccess);
  TestZeroPointSupport<ZeroPointSupport::kGeneral>(
      SymmetricZeroPoint<LhsScalar>() - 1, SymmetricZeroPoint<RhsScalar>(),
      SymmetricZeroPoint<DstScalar>(), ExpectedOutcome::kSuccess);
  TestZeroPointSupport<ZeroPointSupport::kSymmetric>(
      SymmetricZeroPoint<LhsScalar>(), SymmetricZeroPoint<RhsScalar>(),
      SymmetricZeroPoint<DstScalar>(), ExpectedOutcome::kSuccess);
  TestZeroPointSupport<ZeroPointSupport::kSymmetric>(
      SymmetricZeroPoint<LhsScalar>() + 1, SymmetricZeroPoint<RhsScalar>(),
      SymmetricZeroPoint<DstScalar>(), ExpectedOutcome::kDeath);
  TestZeroPointSupport<ZeroPointSupport::kSymmetric>(
      SymmetricZeroPoint<LhsScalar>(), SymmetricZeroPoint<RhsScalar>() + 1,
      SymmetricZeroPoint<DstScalar>(), ExpectedOutcome::kDeath);
  TestZeroPointSupport<ZeroPointSupport::kSymmetric>(
      SymmetricZeroPoint<LhsScalar>(), SymmetricZeroPoint<RhsScalar>(),
      SymmetricZeroPoint<DstScalar>() - 1, ExpectedOutcome::kDeath);
}

TEST(TestSpecialSpecs, PackedLinearRCC) {
  using PackedLinearRCCSpec = PackedLinearRCCSpec<AccumScalar, DstScalar>;
  using PackedLinearRCCTestSet =
      TestSet<LhsScalar, RhsScalar, PackedLinearRCCSpec>;
  TestPackedLinearRCC<PackedLinearRCCTestSet>(81, 93, 72);
  TestLinearAllOrders<PackedLinearRCCTestSet>(81, 93, 72,
                                              ExpectedOutcome::kDeath);
}

}  // namespace ruy