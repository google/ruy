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
project('ruy', 'cpp', default_options : ['default_library=static',
                                         'buildtype=debugoptimized',
                                         'warning_level=1',
                                         'cpp_std=gnu++14'])

if not meson.is_subproject() 
  cc = meson.get_compiler('cpp')
endif

include = include_directories('.')
cpuinfo_dep = dependency('cpuinfo', fallback : ['cpuinfo', 'cpuinfo_dep'])

# We are not using subdir() because this cause meson to include the 
# sub-directory to includes and make 
# accessible with <time.h> which hide the system time.h.

ruy_sources = files(
  'ruy/allocator.cc',
  'ruy/apply_multiplier.cc',
  'ruy/block_map.cc',
  'ruy/blocking_counter.cc',
  'ruy/context.cc',
  'ruy/context_get_ctx.cc',
  'ruy/cpuinfo.cc',
  'ruy/ctx.cc',
  'ruy/denormal.cc',
  'ruy/frontend.cc',
  'ruy/have_built_path_for_avx.cc',
  'ruy/have_built_path_for_avx2_fma.cc',
  'ruy/have_built_path_for_avx512.cc',
  'ruy/kernel_avx.cc',
  'ruy/kernel_avx2_fma.cc',
  'ruy/kernel_avx512.cc',
  'ruy/pack_avx.cc',
  'ruy/pack_avx2_fma.cc',
  'ruy/pack_avx512.cc',
  'ruy/pmu.cc',
  'ruy/prepacked_cache.cc',
  'ruy/prepare_packed_matrices.cc',
  'ruy/system_aligned_alloc.cc',
  'ruy/thread_pool.cc',
  'ruy/trmul.cc',
  'ruy/tune.cc',
  'ruy/wait.cc',
)

ruy_test_sources = files(
    'ruy/allocator_test.cc',
    'ruy/apply_multiplier_test.cc',
    'ruy/block_map_test.cc',
    'ruy/check_macros_test.cc',
    'ruy/context_test.cc',
    'ruy/ctx_test.cc',
    'ruy/gtest_wrapper.h',
    'ruy/matrix_test.cc',
    'ruy/mul_params_test.cc',
    'ruy/prepacked_cache_test.cc',
    'ruy/size_util_test.cc',
    'ruy/test.h',
    'ruy/test_fast.cc',
    'ruy/test_overflow_dst_zero_point.cc',
    'ruy/test_slow.cc',
    'ruy/tune_test.cc',
    'ruy/wait_test.cc',
)

ruy_arm_sources= files(
  'ruy/kernel_arm32.cc',
  'ruy/kernel_arm64.cc',
  'ruy/pack_arm.cc'
)

ruy_profiler_sources = files(
   'ruy/profiler/instrumentation.cc',
   'ruy/profiler/profiler.cc',
)

ruy_profiler_test_sources = files(
   'ruy/profiler/test.cc',
   'ruy/profiler/test_instrumented_library.cc'
)

ruy_sources += ruy_profiler_sources

ruy = library('ruy',
  ruy_sources,
  include_directories : [include],
  install : true,
  dependencies : [cpuinfo_dep],
  cpp_args : ['-mavx', '-DCPUINFO_SUPPORTED_PLATFORM=1'],
  link_args : [],
)

ruy_dep = declare_dependency(
  link_with : ruy,
  include_directories : [include]
)
