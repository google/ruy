# Copyright 2021 Google LLC
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

# Do not edit! To regenerate, run:
#   cmake/bazel_to_cmake.sh


cmake_policy(SET CMP0012 NEW)
cmake_policy(SET CMP0048 NEW)
project(ruy CXX)
cmake_minimum_required(VERSION 3.13)  # Copied from IREE
set(CMAKE_CXX_STANDARD 14)

option(RUY_ENABLE_TESTS "Enable ruy's tests" ON)
if (RUY_ENABLE_TESTS)
  enable_testing()
endif()

option(RUY_PROFILER "Enable ruy's built-in profiler (harms performance)" OFF)

include(cmake/ruy_add_all_subdirs.cmake)
include(cmake/ruy_cc_library.cmake)
include(cmake/ruy_cc_binary.cmake)
include(cmake/ruy_cc_test.cmake)

# Disabling cpuinfo's tests and benchmarks to prevent a copy of its
# googletest dependency getting downloaded into a 'deps' directory in the
# source tree!
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
include_directories("third_party/cpuinfo")
include_directories("third_party/googletest/googletest")
    

ruy_add_all_subdirs()
