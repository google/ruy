"""Build definitions for Ruy."""

# Helper for ruy_copts().
# Returns warnings flags to use for all ruy code.
def ruy_copts_warnings():
    # TensorFlow is C++14 at the moment.
    language_version_warnings = ["-Wc++14-compat"]
    return language_version_warnings + select({
        # We run into trouble on some Windows clang toolchains with -Wextra.
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "-Wall",
            "-Wextra",
        ],
    })

# Helper for ruy_copts().
# Returns flags to use to enable NEON if applicable, for all ruy code.
def ruy_copts_neon():
    return select({
        # OK to crash old devices that lack full NEON support.
        # No need to pass -mfloat-abi=softfp, that is already on.
        "//ruy:armeabi-v7a": [
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    })

# Helper for ruy_copts().
# Returns optimization flags to use for all ruy code.
def ruy_copts_optimize():
    return select({
        # On some toolchains, typically mobile, "-c opt" is interpreted by
        # default as "optimize for size, not for speed". For Ruy code,
        # optimizing for speed is the better compromise, so we override that.
        # Careful to keep debug builds debuggable, whence the select based
        # on the compilation mode.
        "//ruy:optimized": ["-O3"],
        "//conditions:default": [],
    })

# Returns compiler flags to use for all ruy code.
def ruy_copts():
    return ruy_copts_warnings() + ruy_copts_neon() + ruy_copts_optimize()

def ruy_copts_avx2():
    return select({
        "//ruy:x86_64": ["-mavx2", "-mfma"],
        "//conditions:default": [],
    })

# TODO(b/147376783): SSE 4.2 support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
def ruy_copts_sse42():
    return select({
        "//ruy:x86_64": ["-msse4.2"],
        "//conditions:default": [],
    })
