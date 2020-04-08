"""Allows to specialize the ruy BUILD to availability of external libraries"""

def ruy_test_ext_defines():
    return select({
        "//tools/cc_target_os:windows": [],
        "//tools/cc_target_os:wasm": [],
        "//tools/cc_target_os:chromiumos": ["RUY_TESTING_ON_CHROMIUMOS"],
        "//conditions:default": ["RUY_TEST_EXTERNAL_PATHS"],
    })

def ruy_test_ext_deps():
    return select({
        "//tools/cc_target_os:windows": [],
        "//conditions:default": [
            "//third_party/eigen3",
            "//third_party/gemmlowp",
            "//third_party/lapack:blas",
        ],
    })
