# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Include guard: safe to include from multiple CMakeLists.txt directories.
if(DEFINED _XQA_CUBINS_CMAKE_INCLUDED)
  return()
endif()
set(_XQA_CUBINS_CMAKE_INCLUDED TRUE)

function(_edgellm_xqa_get_required_sms OUT_VAR)
  set(_all_sm_versions
      80
      86
      87
      89
      100
      101
      120
      121)
  set(_required_sm_versions "")

  foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
    if(_arch STREQUAL "all"
       OR _arch STREQUAL "all-major"
       OR _arch STREQUAL "native")
      set(_required_sm_versions ${_all_sm_versions})
      break()
    endif()

    string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
    if(_arch_num STREQUAL "")
      continue()
    endif()

    if(_arch_num STREQUAL "110")
      set(_arch_num 101)
    endif()

    if(_arch_num IN_LIST _all_sm_versions)
      list(APPEND _required_sm_versions ${_arch_num})
    endif()
  endforeach()

  # Always include the native GPU SM so decode XQA kernels match the running
  # hardware even when CMAKE_CUDA_ARCHITECTURES omits it (e.g. FMHA pre-built
  # cubins may be broken for sm_87, so the project uses 80;86;89 for FMHA while
  # XQA must still generate a native sm_87 cubin for Orin).
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    RESULT_VARIABLE _nvsmi_rc
    OUTPUT_VARIABLE _native_sm_raw
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_nvsmi_rc EQUAL 0)
    string(STRIP "${_native_sm_raw}" _native_sm_str)
    string(REPLACE "." "" _native_sm "${_native_sm_str}")
    if(_native_sm IN_LIST _all_sm_versions AND NOT _native_sm IN_LIST _required_sm_versions)
      list(APPEND _required_sm_versions ${_native_sm})
      message(STATUS "XQA Kernels: appending native SM ${_native_sm} to cubin generation")
    endif()
  endif()

  list(REMOVE_DUPLICATES _required_sm_versions)
  set(${OUT_VAR}
      "${_required_sm_versions}"
      PARENT_SCOPE)
endfunction()

function(edgellm_xqa_cubins_setup)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  _edgellm_xqa_get_required_sms(_required_sm_versions)

  if(NOT _required_sm_versions)
    message(
      FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}' does not contain a supported XQA SM architecture."
    )
  endif()

  set(_cubin_generator "${CMAKE_SOURCE_DIR}/kernelSrcs/xqa/gen_cubins.py")
  set(_kernel_source_dir "${CMAKE_SOURCE_DIR}/kernelSrcs/xqa")
  set(_generated_cubin_dir "${CMAKE_BINARY_DIR}/generated/xqa/cubin")
  file(GLOB _cubin_inputs "${_kernel_source_dir}/*.cu"
       "${_kernel_source_dir}/*.h" "${_kernel_source_dir}/*.cuh")
  list(APPEND _cubin_inputs "${_cubin_generator}")

  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} ${_cubin_generator} --cmake-invoked --output-dir
      ${_generated_cubin_dir} --arches ${_required_sm_versions} --nvcc
      ${CMAKE_CUDA_COMPILER} --list-outputs
    WORKING_DIRECTORY ${_kernel_source_dir}
    RESULT_VARIABLE _list_outputs_result
    OUTPUT_VARIABLE _generated_outputs_raw
    ERROR_VARIABLE _list_outputs_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _list_outputs_result EQUAL 0)
    message(
      FATAL_ERROR
        "Failed to enumerate generated XQA cubin outputs: ${_list_outputs_error}"
    )
  endif()

  set(_generated_outputs ${_generated_outputs_raw})
  set(_generated_cubin_srcs ${_generated_outputs})
  list(FILTER _generated_cubin_srcs INCLUDE REGEX "\\.cubin\\.cpp$")
  set_source_files_properties(${_generated_outputs} PROPERTIES GENERATED TRUE)

  add_custom_command(
    OUTPUT ${_generated_outputs}
    COMMAND
      ${Python3_EXECUTABLE} ${_cubin_generator} --cmake-invoked --output-dir
      ${_generated_cubin_dir} --arches ${_required_sm_versions} --nvcc
      ${CMAKE_CUDA_COMPILER}
    WORKING_DIRECTORY ${_kernel_source_dir}
    DEPENDS ${_cubin_inputs}
    COMMENT
      "Generating XQA cubins for SM architectures: ${_required_sm_versions}"
    VERBATIM)
  add_custom_target(generateXQACubins DEPENDS ${_generated_outputs})
  message(
    STATUS
      "XQA Kernels: generating cubins for SM architectures: ${_required_sm_versions}"
  )

  set(XQA_CUBIN_INCLUDE_DIR
      "${_generated_cubin_dir}"
      PARENT_SCOPE)
  set(XQA_GENERATED_CUBIN_SRCS
      "${_generated_cubin_srcs}"
      PARENT_SCOPE)
endfunction()

function(edgellm_xqa_add_generated_sources SOURCE_LIST_VAR)
  set(_sources ${${SOURCE_LIST_VAR}})
  list(FILTER _sources EXCLUDE REGEX
       "decodeAttentionKernels/cubin/.*\\.cubin\\.cpp$")
  list(APPEND _sources ${XQA_GENERATED_CUBIN_SRCS})
  set(${SOURCE_LIST_VAR}
      "${_sources}"
      PARENT_SCOPE)
endfunction()

function(edgellm_xqa_configure_target TARGET_NAME)
  target_include_directories(${TARGET_NAME} PRIVATE ${XQA_CUBIN_INCLUDE_DIR})
  add_dependencies(${TARGET_NAME} generateXQACubins)
endfunction()
