# Provides the ruy_test macro for type-parametrized tests.
"""ruy_test is a macro for building a test with multiple paths corresponding to tuples of types for LHS, RHS, accumulator and destination."""

load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

def ruy_test(name, srcs, lhs_rhs_accum_dst, copts, tags = [], deps = None):
    for (lhs, rhs, accum, dst) in lhs_rhs_accum_dst:
        cc_test(
            name = "%s_%s_%s_%s_%s" % (name, lhs, rhs, accum, dst),
            srcs = srcs,
            copts = copts + [
                "-DRUY_TEST_LHSSCALAR=%s" % lhs,
                "-DRUY_TEST_RHSSCALAR=%s" % rhs,
                "-DRUY_TEST_ACCUMSCALAR=%s" % accum,
                "-DRUY_TEST_DSTSCALAR=%s" % dst,
            ],
            deps = deps,
            tags = tags,
        )

def ruy_benchmark(name, srcs, lhs_rhs_accum_dst, copts, deps = None):
    tags = ["req_dep=//third_party/gemmlowp:profiler"]
    for (lhs, rhs, accum, dst) in lhs_rhs_accum_dst:
        cc_binary(
            name = "%s_%s_%s_%s_%s" % (name, lhs, rhs, accum, dst),
            testonly = True,
            srcs = srcs,
            copts = copts + [
                "-DRUY_TEST_LHSSCALAR=%s" % lhs,
                "-DRUY_TEST_RHSSCALAR=%s" % rhs,
                "-DRUY_TEST_ACCUMSCALAR=%s" % accum,
                "-DRUY_TEST_DSTSCALAR=%s" % dst,
            ],
            deps = deps,
            tags = tags,
        )
