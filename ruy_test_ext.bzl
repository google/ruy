"""
Allows to specialize the ruy BUILD to availability of external libraries
"""

def ruy_test_ext_defines():
    return ["RUY_TEST_EXTERNAL_PATHS"]

def ruy_test_ext_deps():
    return [
        "//third_party/eigen3",
        "//third_party/gemmlowp",
        "//third_party/lapack:blas",
    ]
