"""Build definitions for Ruy.

In some cases these are used to configure specific targets for
specific platforms, and dispatch is based on runtime capability detection.
"""

# 1. Enable -mfpu=neon unconditionally on ARM32. If it turns out that we need to support
#    ARM32 without NEON then we'll implement runtime detection and dispatch at that point.
# 2. Explicitly pass -O3 on optimization configs where just "-c opt" means "optimize for code size".

def ruy_copts_base():
    return select({
        ":armeabi-v7a": [
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    }) + select({
        ":optimized": ["-O3"],
        "//conditions:default": [],
    })

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_skylake():
    return select({
        ":x86_64": ["-march=skylake-avx512"],
        "//conditions:default": [],
    })

# Used for targets that are compiled with extra features that are skipped at runtime if unavailable.
def ruy_copts_avx2():
    return select({
        ":x86_64": ["-mavx2", "-mfma"],
        "//conditions:default": [],
    })

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
    return select({
        # TODO(b/146494398): Reinstate flag, something like "-march=cascadelake".
        ":x86_64": [],
        "//conditions:default": [],
    })
