#!/bin/bash

this_script_dir="$(dirname "$0")"

root_dir="$(git -C "${this_script_dir}" rev-parse --show-toplevel)"

build_files="$(find "${root_dir}" -type f -name BUILD)"

if ! command -v python3 &> /dev/null
then
  python_command=python
else
  python_command=python3
fi

for build_file in ${build_files}
do
    package_dir="$(dirname "${build_file}")"
    "${python_command}" "${this_script_dir}/bazel_to_cmake.py" "${root_dir}" "${package_dir}" > "${package_dir}/CMakeLists.txt"
done