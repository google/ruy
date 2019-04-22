"""
Control of ruy visibility
"""

def ruy_visibility():
    return [
        "//third_party/mlir_edge:__subpackages__",
        "//third_party/XNNPACK:__subpackages__",
        "//third_party/tensorflow/lite/kernels:__subpackages__",
    ]
