# Ruy is not BLAS

load("@bazel_skylib//lib:selects.bzl", "selects")
load(":build_defs.bzl", "ruy_copts", "ruy_copts_avx2", "ruy_copts_avx512", "ruy_copts_sse42")
load(":build_defs.oss.bzl", "ruy_copts_avxvnni", "ruy_linkopts_thread_standard_library")
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
    name = "armv7a",
    values = {"cpu": "armv7a"},
)

# Detect ARM 32-bit targets where we are going to just assume NEON support.
selects.config_setting_group(
    name = "arm32_assuming_neon",
    match_any = [
        ":armeabi-v7a",
        ":armv7a",
    ],
)

config_setting(
    name = "x86_64_k8",
    values = {"cpu": "k8"},
)

config_setting(
    name = "x86_64_haswell",
    values = {"cpu": "haswell"},
)

selects.config_setting_group(
    name = "x86_64",
    match_any = [
        ":x86_64_k8",
        ":x86_64_haswell",
    ],
)

config_setting(
    name = "ppc",
    values = {
        "cpu": "ppc",
    },
)

config_setting(
    name = "fuchsia",
    values = {"cpu": "fuchsia"},
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
    copts = ruy_copts(),
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
    copts = ruy_copts(),
)

cc_test(
    name = "check_macros_test",
    srcs = ["check_macros_test.cc"],
    copts = ruy_copts(),
    deps = [
        ":check_macros",
        ":gtest_wrapper",
    ],
)

cc_library(
    name = "opt_set",
    hdrs = ["opt_set.h"],
    copts = ruy_copts(),
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    copts = ruy_copts(),
)

cc_library(
    name = "wait",
    srcs = ["wait.cc"],
    hdrs = ["wait.h"],
    copts = ruy_copts(),
    linkopts = ruy_linkopts_thread_standard_library(),
    deps = [":time"],
)

cc_test(
    name = "wait_test",
    srcs = ["wait_test.cc"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [":check_macros"],
)

cc_test(
    name = "size_util_test",
    srcs = ["size_util_test.cc"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [
        ":opt_set",
        ":platform",
        ":time",
    ],
)

cc_library(
    name = "system_aligned_alloc",
    srcs = [
        "system_aligned_alloc.cc",
    ],
    hdrs = [
        "system_aligned_alloc.h",
    ],
    copts = ruy_copts(),
)

cc_library(
    name = "prepacked_cache",
    srcs = [
        "prepacked_cache.cc",
    ],
    hdrs = [
        "prepacked_cache.h",
    ],
    copts = ruy_copts(),
    deps = [
        ":mat",
        ":system_aligned_alloc",
        "//ruy/profiler:instrumentation",
    ],
)

cc_test(
    name = "tune_test",
    srcs = ["tune_test.cc"],
    copts = ruy_copts(),
    deps = [
        ":gtest_wrapper",
        ":tune",
    ],
)

cc_test(
    name = "prepacked_cache_test",
    srcs = ["prepacked_cache_test.cc"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [
        ":size_util",
        ":system_aligned_alloc",
    ],
)

cc_test(
    name = "allocator_test",
    srcs = ["allocator_test.cc"],
    copts = ruy_copts(),
    deps = [
        ":allocator",
        ":gtest_wrapper",
    ],
)

cc_library(
    name = "side_pair",
    hdrs = ["side_pair.h"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [
        ":check_macros",
        ":cpu_cache_params",
        ":opt_set",
        ":side_pair",
        ":size_util",
        "//ruy/profiler:instrumentation",
    ],
)

cc_test(
    name = "block_map_test",
    srcs = ["block_map_test.cc"],
    copts = ruy_copts(),
    deps = [
        ":block_map",
        ":cpu_cache_params",
        ":gtest_wrapper",
        ":path",
        ":platform",
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
    copts = ruy_copts(),
    linkopts = ruy_linkopts_thread_standard_library(),
    deps = [
        ":check_macros",
        ":time",
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
    copts = ruy_copts(),
    linkopts = ruy_linkopts_thread_standard_library(),
    visibility = ["//visibility:public"],
    deps = [
        ":blocking_counter",
        ":check_macros",
        ":time",
        ":wait",
    ],
)

cc_library(
    name = "cpu_cache_params",
    hdrs = ["cpu_cache_params.h"],
    copts = ruy_copts(),
)

cc_library(
    name = "cpuinfo",
    srcs = [
        "cpuinfo.cc",
    ],
    hdrs = [
        "cpuinfo.h",
    ],
    copts = ruy_copts() +
            select({
                "@bazel_tools//src/conditions:windows": [],
                "//conditions:default": [
                    # ruy_copts contains -Wundef, but cpuinfo's header warns with that.
                    "-Wno-undef",
                ],
            }) + select({
        # This select must match the similar select in `deps`.
        # We intentionally define this token in the BUILD
        # file so that ports to other build-systems do not
        # use cpuinfo by default - they need to port the
        # cpuinfo BUILD first, then can define this token.
        ":ppc": [],
        ":fuchsia": [],
        "//conditions:default": ["-DRUY_HAVE_CPUINFO"],
    }),
    deps = [
        ":platform",
        ":check_macros",
        ":cpu_cache_params",
    ] + select({
        # This select must match the similar select in `copts`
        ":ppc": [],
        ":fuchsia": [],
        "//conditions:default": ["@cpuinfo//:cpuinfo_with_unstripped_include_path"],
    }),
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    copts = ruy_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":size_util",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    copts = ruy_copts(),
    visibility = ["//visibility:public"],
    deps = [":check_macros"],
)

cc_test(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    copts = ruy_copts(),
    deps = [
        ":gtest_wrapper",
        ":matrix",
    ],
)

cc_library(
    name = "mul_params",
    hdrs = ["mul_params.h"],
    copts = ruy_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":matrix",
    ],
)

cc_test(
    name = "mul_params_test",
    srcs = ["mul_params_test.cc"],
    copts = ruy_copts(),
    deps = [
        ":gtest_wrapper",
        ":mul_params",
    ],
)

cc_library(
    name = "mat",
    hdrs = ["mat.h"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [
        ":opt_set",
    ],
)

cc_library(
    name = "apply_multiplier",
    srcs = ["apply_multiplier.cc"],
    hdrs = ["apply_multiplier.h"],
    copts = ruy_copts(),
)

cc_library(
    name = "kernel_common",
    hdrs = [
        "kernel.h",
        "kernel_arm.h",
        "kernel_common.h",
        "kernel_x86.h",
    ],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
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
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [
        ":common",
        ":opt_set",
        ":pack_common",
        ":platform",
        "//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "kernel_avx512",
    srcs = [
        "kernel_avx512.cc",
    ],
    copts = ruy_copts() + ruy_copts_avx512(),
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
    copts = ruy_copts() + ruy_copts_avx512(),
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
    copts = ruy_copts() + ruy_copts_avx512(),
    deps = [
        ":opt_set",
        ":platform",
    ],
)

cc_library(
    name = "kernel_avx2",
    srcs = [
        "kernel_avx2.cc",
    ],
    copts = ruy_copts() + ruy_copts_avx2(),
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
    copts = ruy_copts() + ruy_copts_avx2(),
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
    copts = ruy_copts() + ruy_copts_avx2(),
    deps = [
        ":opt_set",
        ":platform",
    ],
)

# TODO(b/147376783): SSE 4.2 support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
cc_library(
    name = "kernel_sse42",
    srcs = [
        "kernel_sse42.cc",
    ],
    copts = ruy_copts() + ruy_copts_sse42(),
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
    copts = ruy_copts() + ruy_copts_sse42(),
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
    copts = ruy_copts() + ruy_copts_sse42(),
    deps = [
        ":opt_set",
        ":platform",
    ],
)

# TODO(b/147376783): AVX-VNNI support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
cc_library(
    name = "kernel_avxvnni",
    srcs = [
        "kernel_avxvnni.cc",
    ],
    copts = ruy_copts() + ruy_copts_avxvnni(),
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
    copts = ruy_copts() + ruy_copts_avxvnni(),
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
    copts = ruy_copts() + ruy_copts_avxvnni(),
    deps = [
        ":opt_set",
        ":platform",
    ],
)

cc_library(
    name = "kernel",
    hdrs = [
        "kernel.h",
        "kernel_common.h",
    ],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":allocator",
        ":check_macros",
        ":ctx",
        ":have_built_path_for",
        ":path",
        ":platform",
        ":prepacked_cache",
        ":thread_pool",
        ":tune",
    ],
)

cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    copts = ruy_copts(),
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
    name = "ctx_header_only_should_not_include_other_ruy_headers",
    testonly = True,
    hdrs = [
        "ctx.h",
    ],
    # Intentionally no deps. This will cause the stand-alone build of ctx.h to
    # fail if ctx.h #includes any other ruy header.
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
    copts = ruy_copts(),
    deps = [
        ":allocator",
        ":check_macros",
        ":cpuinfo",
        ":have_built_path_for",
        ":path",
        ":platform",
        ":prepacked_cache",
        ":thread_pool",
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
    copts = ruy_copts(),
    deps = [
        ":context",
        ":ctx",
    ],
)

cc_test(
    name = "ctx_test",
    srcs = ["ctx_test.cc"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [
        ":mat",
        ":path",
        ":side_pair",
        ":tune",
    ],
)

cc_library(
    name = "trmul",
    srcs = ["trmul.cc"],
    hdrs = ["trmul.h"],
    copts = ruy_copts(),
    deps = [
        ":allocator",
        ":block_map",
        ":check_macros",
        ":common",
        ":cpu_cache_params",
        ":cpuinfo",
        ":ctx",
        ":mat",
        ":matrix",
        ":mul_params",
        ":opt_set",
        ":side_pair",
        ":size_util",
        ":thread_pool",
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
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    deps = [":check_macros"],
)

cc_library(
    name = "reference_mul",
    hdrs = ["reference_mul.h"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
    # need defines, not copts, because it's controlling a header, test.h
    defines = ruy_test_ext_defines(),
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":allocator",
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
        "//ruy/profiler",
    ] + ruy_test_ext_deps(),
)

ruy_benchmark(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    copts = ruy_copts(),
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
    copts = ruy_copts(),
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
        "//ruy:test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

ruy_test(
    name = "test_slow",
    srcs = ["test_slow.cc"],
    copts = ruy_copts(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("i8", "i8", "i32", "i8"),
        ("u8", "u8", "i32", "i16"),
        ("i8", "i8", "i32", "i32"),
    ],
    tags = ["slow"],
    deps = [
        "//ruy:test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

ruy_test(
    name = "test_special_mul_params",
    srcs = ["test_special_mul_params.cc"],
    copts = ruy_copts(),
    lhs_rhs_accum_dst = [
        ("f32", "f32", "f32", "f32"),
        ("u8", "u8", "i32", "u8"),
        ("u8", "u8", "i32", "i16"),
    ],
    deps = [
        "//ruy:test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
