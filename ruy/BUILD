# Ruy is not BLAS

load(":build_defs.oss.bzl", "ruy_copts_avx2", "ruy_copts_avxvnni", "ruy_copts_base", "ruy_copts_skylake", "ruy_copts_sse42", "ruy_linkopts_thread_standard_library")
load(":ruy_test_ext.oss.bzl", "ruy_test_ext_defines", "ruy_test_ext_deps")
load(":ruy_test.bzl", "ruy_benchmark", "ruy_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "armeabi-v7a",
    values = {"cpu": "armeabi-v7a"},
)

config_setting(
    name = "x86_64",
    values = {"cpu": "k8"},
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "gtest_wrapper",
    testonly = True,
    hdrs = ["gtest_wrapper.h"],
    visibility = [":__subpackages__"],
    deps = ["@com_google_googletest//:gtest"],
)

cc_library(
    name = "check_macros",
    hdrs = ["check_macros.h"],
    copts = ruy_copts_base(),
)

cc_test(
    name = "check_macros_test",
    srcs = ["check_macros_test.cc"],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":gtest_wrapper",
    ],
)

cc_library(
    name = "opt_set",
    hdrs = ["opt_set.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "wait",
    srcs = ["wait.cc"],
    hdrs = ["wait.h"],
    copts = ruy_copts_base(),
    linkopts = ruy_linkopts_thread_standard_library(),
    deps = [":time"],
)

cc_test(
    name = "wait_test",
    srcs = ["wait_test.cc"],
    linkopts = ruy_linkopts_thread_standard_library(),
    deps = [
        ":gtest_wrapper",
        ":platform",
        ":wait",
    ],
)

cc_library(
    name = "size_util",
    hdrs = ["size_util.h"],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [":check_macros"],
)

cc_test(
    name = "size_util_test",
    srcs = ["size_util_test.cc"],
    deps = [
        ":gtest_wrapper",
        ":size_util",
    ],
)

cc_library(
    name = "tune",
    srcs = [
        "tune.cc",
    ],
    hdrs = [
        "tune.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":opt_set",
        ":platform",
        ":time",
    ],
)

cc_library(
    name = "prepacked_cache",
    srcs = [
        "prepacked_cache.cc",
    ],
    hdrs = [
        "prepacked_cache.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":allocator",
        ":mat",
        ":opt_set",
        ":platform",
        ":time",
        "//ruy/profiler:instrumentation",
    ],
)

cc_test(
    name = "tune_test",
    srcs = ["tune_test.cc"],
    deps = [
        ":gtest_wrapper",
        ":tune",
    ],
)

cc_test(
    name = "prepacked_cache_test",
    srcs = ["prepacked_cache_test.cc"],
    deps = [
        ":context",
        ":context_get_ctx",
        ":ctx",
        ":gtest_wrapper",
        ":mat",
        ":matrix",
        ":prepacked_cache",
        ":ruy",
        ":time",
    ],
)

cc_binary(
    name = "tune_tool",
    srcs = ["tune_tool.cc"],
    deps = [
        ":tune",
    ],
)

cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc",
    ],
    hdrs = [
        "allocator.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":size_util",
    ],
)

cc_test(
    name = "allocator_test",
    srcs = ["allocator_test.cc"],
    deps = [
        ":allocator",
        ":gtest_wrapper",
    ],
)

cc_library(
    name = "side_pair",
    hdrs = ["side_pair.h"],
    copts = ruy_copts_base(),
    deps = [":check_macros"],
)

cc_library(
    name = "block_map",
    srcs = [
        "block_map.cc",
    ],
    hdrs = [
        "block_map.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":opt_set",
        ":side_pair",
        ":size_util",
        "//ruy/profiler:instrumentation",
    ],
)

cc_test(
    name = "block_map_test",
    srcs = ["block_map_test.cc"],
    deps = [
        ":block_map",
        ":cpu_cache_size",
        ":gtest_wrapper",
        ":path",
        ":side_pair",
    ],
)

cc_library(
    name = "blocking_counter",
    srcs = [
        "blocking_counter.cc",
    ],
    hdrs = [
        "blocking_counter.h",
    ],
    copts = ruy_copts_base(),
    linkopts = ruy_linkopts_thread_standard_library(),
    deps = [
        ":check_macros",
        ":wait",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = [
        "thread_pool.cc",
    ],
    hdrs = [
        "thread_pool.h",
    ],
    copts = ruy_copts_base(),
    linkopts = ruy_linkopts_thread_standard_library(),
    visibility = ["//visibility:public"],
    deps = [
        ":blocking_counter",
        ":check_macros",
        ":wait",
    ],
)

cc_library(
    name = "detect_arm",
    srcs = [
        "detect_arm.cc",
    ],
    hdrs = [
        "detect_arm.h",
    ],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "detect_x86",
    srcs = [
        "detect_x86.cc",
    ],
    hdrs = [
        "detect_x86.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":platform",
    ],
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":size_util",
    ],
)

cc_library(
    name = "cpu_cache_size",
    hdrs = ["cpu_cache_size.h"],
    copts = ruy_copts_base(),
    deps = [
        ":path",
        ":platform",
    ],
)

cc_library(
    name = "trace",
    srcs = [
        "trace.cc",
    ],
    hdrs = [
        "trace.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":block_map",
        ":check_macros",
        ":side_pair",
        ":time",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [":check_macros"],
)

cc_test(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    deps = [
        ":gtest_wrapper",
        ":matrix",
    ],
)

cc_library(
    name = "mul_params",
    hdrs = ["mul_params.h"],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [
        ":cpu_cache_size",
        ":matrix",
    ],
)

cc_test(
    name = "mul_params_test",
    srcs = ["mul_params_test.cc"],
    deps = [
        ":gtest_wrapper",
        ":mul_params",
    ],
)

cc_library(
    name = "mat",
    hdrs = ["mat.h"],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":common",
        ":matrix",
        ":size_util",
    ],
)

cc_library(
    name = "common",
    hdrs = [
        "common.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":path",
        ":platform",
    ],
)

cc_library(
    name = "apply_multiplier",
    srcs = ["apply_multiplier.cc"],
    hdrs = ["apply_multiplier.h"],
    copts = ruy_copts_base(),
)

cc_library(
    name = "kernel_common",
    hdrs = [
        "kernel.h",
        "kernel_arm.h",
        "kernel_common.h",
        "kernel_x86.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":apply_multiplier",
        ":check_macros",
        ":common",
        ":mat",
        ":matrix",
        ":mul_params",
        ":opt_set",
        ":path",
        ":platform",
        ":side_pair",
        ":size_util",
        ":tune",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack_common",
    hdrs = [
        "pack.h",
        "pack_arm.h",
        "pack_common.h",
        "pack_x86.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":common",
        ":mat",
        ":matrix",
        ":opt_set",
        ":path",
        ":platform",
        ":tune",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "kernel_arm",
    srcs = [
        "kernel_arm32.cc",
        "kernel_arm64.cc",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":common",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack_arm",
    srcs = [
        "pack_arm.cc",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":common",
        ":opt_set",
        ":pack_common",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

# AVX-512 compilation units.
#
# These must use the same compiler options.
RUY_COPTS_BUILT_FOR_AVX512 = ruy_copts_base() + ruy_copts_skylake()

cc_library(
    name = "kernel_avx512",
    srcs = [
        "kernel_avx512.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX512,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack_avx512",
    srcs = [
        "pack_avx512.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX512,
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":pack_common",
        ":path",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "have_built_path_for_avx512",
    srcs = [
        "have_built_path_for_avx512.cc",
    ],
    hdrs = [
        "have_built_path_for.h",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX512,
    deps = [
        ":opt_set",
        ":platform",
    ],
)
# End: AVX-512 compilation units.

# AVX2 compilation units.
#
# These must use the same compiler options.
RUY_COPTS_BUILT_FOR_AVX2 = ruy_copts_base() + ruy_copts_avx2()

cc_library(
    name = "kernel_avx2",
    srcs = [
        "kernel_avx2.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX2,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack_avx2",
    srcs = [
        "pack_avx2.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX2,
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":pack_common",
        ":path",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "have_built_path_for_avx2",
    srcs = [
        "have_built_path_for_avx2.cc",
    ],
    hdrs = [
        "have_built_path_for.h",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX2,
    deps = [
        ":opt_set",
        ":platform",
    ],
)
# End: AVX2 compilation units.

# SSE42 compilation units.
#
# TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
#
# These must use the same compiler options.
RUY_COPTS_BUILT_FOR_SSE42 = ruy_copts_base() + ruy_copts_sse42()

cc_library(
    name = "kernel_sse42",
    srcs = [
        "kernel_sse42.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_SSE42,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack_sse42",
    srcs = [
        "pack_sse42.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_SSE42,
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":pack_common",
        ":path",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "have_built_path_for_sse42",
    srcs = [
        "have_built_path_for_sse42.cc",
    ],
    hdrs = [
        "have_built_path_for.h",
    ],
    copts = RUY_COPTS_BUILT_FOR_SSE42,
    deps = [
        ":opt_set",
        ":platform",
    ],
)
# End: SSE42 compilation units.

# AVX-VNNI compilation units.
#
# TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
#
# These must use the same compiler options.
RUY_COPTS_BUILT_FOR_AVX_VNNI = ruy_copts_base() + ruy_copts_avxvnni()

cc_library(
    name = "kernel_avxvnni",
    srcs = [
        "kernel_avxvnni.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX_VNNI,
    deps = [
        ":check_macros",
        ":kernel_common",
        ":opt_set",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack_avxvnni",
    srcs = [
        "pack_avxvnni.cc",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX_VNNI,
    deps = [
        ":check_macros",
        ":matrix",
        ":opt_set",
        ":pack_common",
        ":path",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "have_built_path_for_avxvnni",
    srcs = [
        "have_built_path_for_avxvnni.cc",
    ],
    hdrs = [
        "have_built_path_for.h",
    ],
    copts = RUY_COPTS_BUILT_FOR_AVX_VNNI,
    deps = [
        ":opt_set",
        ":platform",
    ],
)
# End: AVX-VNNI compilation units.

cc_library(
    name = "kernel",
    hdrs = [
        "kernel.h",
        "kernel_common.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":apply_multiplier",
        ":check_macros",
        ":common",
        ":kernel_arm",  # fixdeps: keep
        ":kernel_avx2",  # fixdeps: keep
        ":kernel_avx512",  # fixdeps: keep
        ":kernel_avxvnni",  # fixdeps: keep
        ":kernel_common",
        ":kernel_sse42",  # fixdeps: keep
        ":mat",
        ":matrix",
        ":mul_params",
        ":opt_set",
        ":path",
        ":platform",
        ":side_pair",
        ":size_util",
        ":tune",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "pack",
    hdrs = [
        "pack.h",
        "pack_common.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":check_macros",
        ":common",
        ":mat",
        ":matrix",
        ":opt_set",
        ":pack_arm",  # fixdeps: keep
        ":pack_avx2",  # fixdeps: keep
        ":pack_avx512",  # fixdeps: keep
        ":pack_avxvnni",  # fixdeps: keep
        ":pack_common",
        ":pack_sse42",  # fixdeps: keep
        ":path",
        ":platform",
        ":tune",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "have_built_path_for",
    hdrs = [
        "have_built_path_for.h",
    ],
    deps = [
        ":have_built_path_for_avx2",
        ":have_built_path_for_avx512",
        ":have_built_path_for_avxvnni",
        ":have_built_path_for_sse42",
        ":platform",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = [
        "context.h",
    ],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [
        ":allocator",
        ":check_macros",
        ":ctx",
        ":detect_arm",
        ":detect_x86",
        ":have_built_path_for",
        ":path",
        ":platform",
        ":prepacked_cache",
        ":thread_pool",
        ":trace",
        ":tune",
    ],
)

cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        ":gtest_wrapper",
        ":path",
        ":platform",
        ":prepacked_cache",
        ":tune",
    ],
)

cc_library(
    name = "ctx",
    srcs = [
        "ctx.cc",
    ],
    hdrs = [
        "ctx.h",
        "ctx_impl.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":allocator",
        ":check_macros",
        ":detect_arm",
        ":detect_x86",
        ":have_built_path_for",
        ":path",
        ":platform",
        ":prepacked_cache",
        ":thread_pool",
        ":trace",
        ":tune",
    ],
)

cc_library(
    name = "context_get_ctx",
    srcs = [
        "context_get_ctx.cc",
    ],
    hdrs = [
        "context_get_ctx.h",
    ],
    copts = ruy_copts_base(),
    deps = [
        ":context",
        ":ctx",
    ],
)

cc_test(
    name = "ctx_test",
    srcs = ["ctx_test.cc"],
    deps = [
        ":ctx",
        ":gtest_wrapper",
        ":path",
        ":platform",
    ],
)

cc_library(
    name = "trmul_params",
    hdrs = ["trmul_params.h"],
    copts = ruy_copts_base(),
    deps = [
        ":mat",
        ":side_pair",
        ":tune",
    ],
)

cc_library(
    name = "trmul",
    srcs = ["trmul.cc"],
    hdrs = ["trmul.h"],
    copts = ruy_copts_base(),
    deps = [
        ":allocator",
        ":block_map",
        ":check_macros",
        ":common",
        ":ctx",
        ":mat",
        ":matrix",
        ":mul_params",
        ":opt_set",
        ":side_pair",
        ":size_util",
        ":thread_pool",
        ":trace",
        ":trmul_params",
        ":tune",
        "//ruy/profiler:instrumentation",
    ],
)

# The main library.
cc_library(
    name = "ruy",
    srcs = [
        "dispatch.h",
    ],
    hdrs = [
        "ruy.h",
    ],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [
        ":check_macros",
        ":common",
        ":context",
        ":context_get_ctx",
        ":ctx",
        ":kernel",
        ":mat",
        ":matrix",
        ":mul_params",
        ":opt_set",
        ":pack",
        ":path",
        ":prepacked_cache",
        ":side_pair",
        ":size_util",
        ":trmul",
        ":trmul_params",
        ":tune",
        "//ruy/profiler:instrumentation",
    ],
)

# Usage examples.
cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [":ruy"],
)

# Small library to query PMU counters, for benchmark only
cc_library(
    name = "pmu",
    testonly = True,
    srcs = ["pmu.cc"],
    hdrs = ["pmu.h"],
    copts = ruy_copts_base(),
    deps = [":check_macros"],
)

cc_library(
    name = "reference_mul",
    hdrs = ["reference_mul.h"],
    copts = ruy_copts_base(),
    visibility = ["//visibility:public"],
    deps = [
        ":apply_multiplier",
        ":matrix",
        ":mul_params",
    ],
)

# Testing framework.
cc_library(
    name = "test_lib",
    testonly = True,
    hdrs = ["test.h"],
    copts = ruy_copts_base(),
    # need defines, not copts, because it's controlling a header, test.h
    defines = ruy_test_ext_defines(),
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":reference_mul",
        ":matrix",
        ":pmu",
        ":ruy",
        ":mul_params",
        ":time",
        ":gtest_wrapper",
        ":platform",
        ":context",
        ":ctx",
        ":context_get_ctx",
        "//ruy/profiler:profiler",
    ] + ruy_test_ext_deps(),
)

ruy_benchmark(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    copts = ruy_copts_base(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
    ],
    deps = [
        "//ruy:test_lib",
        "//ruy/profiler:instrumentation",
    ],
)

ruy_test(
    name = "test_fast",
    srcs = ["test_fast.cc"],
    copts = ruy_copts_base(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("f64", "f32", "f64", "f32"),
        ("f32", "f64", "f64", "f64"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("i8", "u8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
        ("i8", "u8", "i32", "i32"),
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//ruy:test_lib",
    ],
)

ruy_test(
    name = "test_slow",
    srcs = ["test_slow.cc"],
    copts = ruy_copts_base(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
    ],
    tags = ["slow"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//ruy:test_lib",
    ],
)

ruy_test(
    name = "test_special_mul_params",
    srcs = ["test_special_mul_params.cc"],
    copts = ruy_copts_base(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("u8", "u8", "i32", "i16"),
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//ruy:test_lib",
    ],
)
