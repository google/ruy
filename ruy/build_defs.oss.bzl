"""Build definitions for Ruy that are specific to the open-source build."""

# TODO(b/147376783): VNNI support is incomplete / placeholder.
# Optimization is not finished. In particular the dimensions of the kernel
# blocks can be changed as desired.
#
# At the moment this does nothing because current toolchains don't support VNNI.
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
