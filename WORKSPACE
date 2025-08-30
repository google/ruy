# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Workspace file for the Ruy project.

workspace(name = "com_google_ruy")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

http_archive(
    name = "bazel_features",
    sha256 = "06a9691becc357c1e4af011b9fb3239097503ce00607d9eb6b11ea0fac304039",
    strip_prefix = "bazel_features-1.35.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.35.0/bazel_features-v1.35.0.tar.gz",
)

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

# skylib utility for additional bazel functionality.
http_archive(
    name = "bazel_skylib",
    sha256 = "51b5105a760b353773f904d2bbc5e664d0987fbaf22265164de65d43e910d8ac",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.8.1/bazel-skylib-1.8.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.8.1/bazel-skylib-1.8.1.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check(minimum_bazel_version = "2.0.0")

http_archive(
    name = "rules_cc",
    sha256 = "e50f24506011841e2ac83d9733a0c7e058eb3a10a6e3e10baa9c7942ff5f4767",
    strip_prefix = "rules_cc-0.2.2",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.2.2/rules_cc-0.2.2.tar.gz",
)

load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")

compatibility_proxy_repo()

http_archive(
    name = "com_google_googletest",
    sha256 = "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c",
    strip_prefix = "googletest-1.17.0",
    url = "https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz",
)

http_archive(
    name = "cpuinfo",
    sha256 = "87fc79472eb30b734cbe700dfe3edc0bdb96c33e3ce44e48ab327bbd8783f07f",
    strip_prefix = "cpuinfo-3c8b1533ac03dd6531ab6e7b9245d488f13a82a5",
    url = "https://github.com/pytorch/cpuinfo/archive/3c8b1533ac03dd6531ab6e7b9245d488f13a82a5.tar.gz",
)
