#!/usr/bin/env python3
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

"""This is yet another bazel-to-cmake converter. It's independently written from
scratch but relies on the same basic idea as others (including IREE's), namely:
to let the python interpreter do the bulk of the work, exploiting the fact that
both Bazel's BUILD syntax and Starlark (".bzl") languages are more or less
subsets of Python.

The main feature that this converter supports and that others don't, justifying
its existence as of early 2021, are
  1. Ad-hoc support for select(), generating CMake if()...elseif()... chains
     parsing the condition keys (e.g. anything ending in ":windows" is interpreted as the condition "the target platform
     is Windows"). This allows to just ignore config_setting, as we only care
     about the config_setting names, not their actual implementation, as well
     as all the variants from the Bazel 'selects' library.
  2. Support for load(), loading macros from Starlark files.
"""

import re
import os
import os.path
import pickle
import sys
import datetime

# Ruy's dependencies.
external_targets = ['gtest', 'gtest_main', 'cpuinfo']

# Text replacements [oldstring, newstring] pairs, applied on all BUILD and
# Starlark files that we load. Only used by preprocess_input_text.
replacements = [
    ['$(STACK_FRAME_UNLIMITED)', ''],
    ['native.cc_', 'cc_'],
    ['selects.config_setting_group', 'config_setting_group'],
    ['@com_google_googletest//:gtest', 'gtest'],
    ['@com_google_googletest//:gtest_main', 'gtest_main'],
    ['@cpuinfo//:cpuinfo_with_unstripped_include_path', 'cpuinfo'],
]


def preprocess_input_text(text):
    result = text
    for replacement in replacements:
        result = result.replace(replacement[0], replacement[1])
    return result


def package(**kwargs):
    pass


def exports_files(*args):
    pass


def load(filename, *args):
    if filename.startswith('@'):
        return
    elif filename.startswith(':'):
        filename = os.path.join(bazel_package_dir, filename[1:])
    elif filename.startswith('//'):
        split = filename[2:].split(':')
        filename = os.path.join(bazel_workspace_dir, split[0], split[1])

    src_file_content = open(filename).read()
    processed_file_content = preprocess_input_text(src_file_content)
    exec(processed_file_content, globals(), globals())


def config_setting(**kwargs):
    pass


def filegroup(**kwargs):
    pass


def config_setting_group(**kwargs):
    pass


def bzl_library(**kwargs):
    pass


def append_cmake_list(list_name, values, indent):
    semicolon_separated = ";".join(values)
    print(f'{indent}set({list_name} "{semicolon_separated}")')


def generate_cmake_select(select_name, dict):
    new_if_branch_keyword = 'if'
    default_value = []
    for key in dict:
        condition = ''
        if key == '//conditions:default':
            default_value = dict[key]
        elif re.search(r':windows$', key):
            condition = 'CMAKE_SYSTEM_NAME STREQUAL Windows'
        elif re.search(r':ppc$', key):
            condition = 'CMAKE_SYSTEM_PROCESSOR STREQUAL ppc64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL ppc64le'
        elif re.search(r':s390x$', key):
            condition = 'CMAKE_SYSTEM_PROCESSOR STREQUAL s390 OR CMAKE_SYSTEM_PROCESSOR STREQUAL s390x'
        elif re.search(r':fuchsia$', key):
            condition = 'CMAKE_SYSTEM_NAME STREQUAL Fuchsia'
        elif re.search(r':arm32_assuming_neon$', key):
            condition = 'CMAKE_SYSTEM_PROCESSOR STREQUAL arm'
        elif re.search(r':do_not_want_O3$', key):
            # This config setting is a work-around for certain Bazel toolchains.
            # Unconditionally returning TRUE here means we will never explicitly
            # set -O3 in CMake, we will rely on CMake's defaults for the given
            # CMake build type.
            condition = 'TRUE'
        elif re.search(r':x86_64_and_not_msvc$', key):
            condition = '(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64) AND NOT MSVC'
        elif re.search(r':windows_msvc$', key):
            condition = 'MSVC'
        elif re.search(r':ruy_profiler$', key):
            condition = '${RUY_PROFILER}'
        else:
            raise ValueError(f'Unhandled key in select: {key}')

        if condition:
            print(f'{new_if_branch_keyword}({condition})')
            if dict[key]:
                append_cmake_list(select_name, dict[key], '  ')
            else:
                print(f'  # nothing goes into {select_name}')
        new_if_branch_keyword = 'elseif'

    print('else()')
    append_cmake_list(select_name, default_value, '  ')

    print('endif()\n')


select_index = 0
select_cache = {}


def select(dict):
    global select_index
    global select_cache
    global package_prefix
    key = pickle.dumps(dict)
    if key in select_cache:
        select_name = select_cache[key]
    else:
        select_name = f'RUY_SELECT_{package_prefix}_{select_index}'
        select_index = select_index + 1
        select_cache[key] = select_name
        generate_cmake_select(select_name, dict)

    return [f'${{{select_name}}}']


def get_cmake_local_target_name(name):
    global package_prefix
    return f'{package_prefix}_{name}'


def get_cmake_dep_target_name(name):
    if name in external_targets:
        return name
    if name.startswith('//'):
        after_last_slash = name.split('/')[-1]
        if not ':' in after_last_slash:
            name = f'{name}:{after_last_slash}'
        return name[2:].replace('/', '_').replace(':', '_')
    if name.startswith(':'):
        name = name[1:]
    if name.startswith('$'):
        return name
    global package_prefix
    return f'{package_prefix}_{name}'


def generic_rule(rule_name, **kwargs):
    print(f'{rule_name}(')
    for key in kwargs.keys():
        if type(kwargs[key]) is bool:
            if kwargs[key]:
                print(f'  {key.upper()}')
                continue
            else:
                raise ValueError(
                    'How do we specify FALSE boolean args in CMake?')
        if key == 'visibility':
            if kwargs[key] == ['//visibility:public']:
                print(f'  PUBLIC')
            continue
        print(f'  {key.upper()}')
        if type(kwargs[key]) is list:
            for value in kwargs[key]:
                if key == 'deps':
                    target_name = get_cmake_dep_target_name(value)
                    print(f'    {target_name}')
                else:
                    print(f'    {value}')
        else:
            if key == 'name':
                target_name = get_cmake_local_target_name(kwargs[key])
                print(f'    {target_name}')
            else:
                print(f'    {kwargs[key]}')
    print(')\n')


def cc_library(**kwargs):
    generic_rule('cc_library', **kwargs)


def cc_test(**kwargs):
    generic_rule('cc_test', **kwargs)


def cc_binary(**kwargs):
    generic_rule('cc_binary', **kwargs)


if len(sys.argv) != 3:
    print("Usage: bazel_to_cmake.py bazel_workspace_dir bazel_package_dir")
    sys.exit(1)

bazel_workspace_dir = sys.argv[1]
bazel_package_dir = sys.argv[2]
bazel_package_relative_dir = os.path.relpath(
    bazel_package_dir, bazel_workspace_dir)
package_prefix = bazel_package_relative_dir.replace(os.path.sep, '_')

print("""# Copyright %d Google LLC
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
""" % datetime.datetime.now().year)

src_build_file = os.path.join(bazel_package_dir, "BUILD")

print("""# Do not edit! This file was auto-generated by this program:
#   %s
# To regenerate CMakeLists.txt for all subdirectories, just run:
#   cmake/bazel_to_cmake.sh
""" % ' '.join([os.path.relpath(p, bazel_workspace_dir) for p in sys.argv]))

if bazel_workspace_dir == bazel_package_dir:
    print("""
cmake_policy(SET CMP0012 NEW)
cmake_policy(SET CMP0048 NEW)
project(ruy CXX)
cmake_minimum_required(VERSION 3.13)  # Copied from IREE
set(CMAKE_CXX_STANDARD 14)

option(RUY_ENABLE_TESTS "Enable ruy's tests" ON)
if (RUY_ENABLE_TESTS)
  enable_testing()
endif()

include(cmake/add_all_subdirs.cmake)
include(cmake/cc_library.cmake)
include(cmake/cc_binary.cmake)
include(cmake/cc_test.cmake)

set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
include_directories("third_party/cpuinfo")
include_directories("third_party/googletest/googletest")
    
""")

src_build_content = open(src_build_file).read()
processed_build_content = preprocess_input_text(src_build_content)
exec(processed_build_content)

print("""

add_all_subdirs()
""")
