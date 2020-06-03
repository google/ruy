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

http_archive(
    name = "com_google_googletest",
    sha256 =
        "9bf1fe5182a604b4135edc1a425ae356c9ad15e9b23f9f12a02e80184c3a249c",
    strip_prefix = "googletest-release-1.8.1",
    urls = [
        "https://github.com/google/googletest/archive/release-1.8.1.tar.gz",
    ],
)

# clog library, used by cpuinfo for logging
http_archive(
    name = "clog",
    strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
    sha256 = "3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz",
    ],
    build_file = "@//third_party:clog.BUILD",
)

# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    strip_prefix = "cpuinfo-c2092219e7c874783a00a62edb94ddc672f57ab3",
    sha256 = "ea56c399a4f6ca5f749e71acb6a7bfdc653eb65d8f658cb2e414a2fcdca1fe8b",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/c2092219e7c874783a00a62edb94ddc672f57ab3.zip",
    ],
    build_file = "@//third_party:cpuinfo.BUILD",
    patches = ["@//third_party:cpuinfo.patch"],
)

# skylib utility for additional bazel functionality.
skylib_version = "0.9.0"
http_archive(
    name = "bazel_skylib",
    type = "tar.gz",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel_skylib-{}.tar.gz".format (skylib_version, skylib_version),
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
)
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "2.0.0")
