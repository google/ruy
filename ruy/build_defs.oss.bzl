"""Build definitions for Ruy."""

# 1. Enable -mfpu=neon unconditionally on ARM32. If it turns out that we need to support
#    ARM32 without NEON then we'll implement runtime detection and dispatch at that point.
# 2. Explicitly pass -O3 on optimization configs where just "-c opt" means "optimize for code size".

def ruy_copts_base():
    return select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-Wall", "-Wextra"],
    }) + select({
        "//ruy:armeabi-v7a": [
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    }) + select({
        "//ruy:optimized": ["-O3"],
        "//conditions:default": [],
    })

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_skylake():
    return []

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_avx2():
    return []

# TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
#
# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_sse42():
    return []

# TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
#
# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_avxvnni():
    return []

# Used for targets that #include <thread>
def ruy_linkopts_thread_standard_library():
    # In open source builds, GCC is a common occurence. It requires "-pthread"
    # to use the C++11 <thread> standard library header. This breaks the
    # opensource build on Windows and probably some other platforms, so that
    # will need to be fixed as needed. Ideally we would like to do this based
    # on GCC being the compiler, but that does not seem to be easy to achieve
    # with Bazel. Instead we do the following, which is copied from
    # https://github.com/abseil/abseil-cpp/blob/1112609635037a32435de7aa70a9188dcb591458/absl/base/BUILD.bazel#L155
    return select({
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": ["-pthread"],
    })
