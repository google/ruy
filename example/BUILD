load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//third_party/ruy:license"],
    licenses = ["notice"],  # Apache 2.0
)

# Usage examples.
cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = ["//ruy"],
)

cc_binary(
    name = "parametrized_example",
    srcs = ["parametrized_example.cc"],
    deps = ["//ruy"],
)
