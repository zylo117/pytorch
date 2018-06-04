##############################################################################
# Macro to update cached options.
macro (caffe2_update_option variable value)
  get_property(__help_string CACHE ${variable} PROPERTY HELPSTRING)
  set(${variable} ${value} CACHE BOOL ${__help_string} FORCE)
endmacro()


##############################################################################
# Add an interface library definition that is dependent on the source.
macro(caffe2_interface_library SRC DST)
  add_library(${DST} INTERFACE)
  add_dependencies(${DST} ${SRC})
  # Depending on the nature of the source library as well as the compiler,
  # determine the needed compilation flags.
  get_target_property(__src_target_type ${SRC} TYPE)
  # Depending on the type of the source library, we will set up the
  # link command for the specific SRC library.
  if (${__src_target_type} STREQUAL "STATIC_LIBRARY")
    # In the case of static library, we will need to add whole-static flags.
    if(APPLE)
      target_link_libraries(
          ${DST} INTERFACE -Wl,-force_load,$<TARGET_FILE:${SRC}>)
    elseif(MSVC)
      # In MSVC, we will add whole archive in default.
      target_link_libraries(
          ${DST} INTERFACE -WHOLEARCHIVE:$<TARGET_FILE:${SRC}>)
    else()
      # Assume everything else is like gcc
      target_link_libraries(${DST} INTERFACE
          "-Wl,--whole-archive,$<TARGET_FILE:${SRC}> -Wl,--no-whole-archive")
    endif()
    # Link all interface link libraries of the src target as well.
    # For static library, we need to explicitly depend on all the libraries
    # that are the dependent library of the source library. Note that we cannot
    # use the populated INTERFACE_LINK_LIBRARIES property, because if one of the
    # dependent library is not a target, cmake creates a $<LINK_ONLY:src> wrapper
    # and then one is not able to find target "src". For more discussions, check
    #   https://gitlab.kitware.com/cmake/cmake/issues/15415
    #   https://cmake.org/pipermail/cmake-developers/2013-May/019019.html
    # Specifically the following quote
    #
    # """
    # For STATIC libraries we can define that the PUBLIC/PRIVATE/INTERFACE keys
    # are ignored for linking and that it always populates both LINK_LIBRARIES
    # LINK_INTERFACE_LIBRARIES.  Note that for STATIC libraries the
    # LINK_LIBRARIES property will not be used for anything except build-order
    # dependencies.
    # """
    target_link_libraries(${DST} INTERFACE
        $<TARGET_PROPERTY:${SRC},LINK_LIBRARIES>)
  elseif(${__src_target_type} STREQUAL "SHARED_LIBRARY")
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
      target_link_libraries(${DST} INTERFACE
          "-Wl,--no-as-needed,$<TARGET_FILE:${SRC}> -Wl,--as-needed")
    else()
      target_link_libraries(${DST} INTERFACE ${SRC})
    endif()
    # Link all interface link libraries of the src target as well.
    # For shared libraries, we can simply depend on the INTERFACE_LINK_LIBRARIES
    # property of the target.
    target_link_libraries(${DST} INTERFACE
        $<TARGET_PROPERTY:${SRC},INTERFACE_LINK_LIBRARIES>)
  else()
    message(FATAL_ERROR
        "You made a CMake build file error: target " ${SRC}
        " must be of type either STATIC_LIBRARY or SHARED_LIBRARY. However, "
        "I got " ${__src_target_type} ".")
  endif()
  # For all other interface properties, manually inherit from the source target.
  set_target_properties(${DST} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_DEFINITIONS>
    INTERFACE_COMPILE_OPTIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_OPTIONS>
    INTERFACE_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_INCLUDE_DIRECTORIES>
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
endmacro()


##############################################################################
# Creating a Caffe2 binary target with sources specified with relative path.
# Usage:
#   caffe2_binary_target(target_name_or_src <src1> [<src2>] [<src3>] ...)
# If only target_name_or_src is specified, this target is build with one single
# source file and the target name is autogen from the filename. Otherwise, the
# target name is given by the first argument and the rest are the source files
# to build the target.
function(caffe2_binary_target target_name_or_src)
  if (${ARGN})
    set(__target ${target_name_or_src})
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${ARGN}")
  else()
    get_filename_component(__target ${target_name_or_src} NAME_WE)
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${target_name_or_src}")
  endif()
  add_executable(${__target} ${__srcs})
  target_link_libraries(${__target} ${Caffe2_MAIN_LIBS})
  # If we have Caffe2_MODULES defined, we will also link with the modules.
  if (DEFINED Caffe2_MODULES)
    target_link_libraries(${__target} ${Caffe2_MODULES})
  endif()
  install(TARGETS ${__target} DESTINATION bin)
endfunction()


##############################################################################
# Multiplex between loading executables for CUDA versus HIP (AMD Software Stack).
# Usage:
#   torch_cuda_based_add_executable(cuda_target)
#
macro(torch_cuda_based_add_executable cuda_target)
  IF (USE_ROCM)
    hip_add_executable(${cuda_target} ${ARGN})
  ELSEIF(USE_CUDA)
    cuda_add_executable(${cuda_target} ${ARGN})
  ELSE()

  ENDIF()
endmacro()


##############################################################################
# Multiplex between adding libraries for CUDA versus HIP (AMD Software Stack).
# Usage:
#   torch_cuda_based_add_library(cuda_target)
#
macro(torch_cuda_based_add_library cuda_target)
  IF (USE_ROCM)
    hip_add_library(${cuda_target} ${ARGN})
  ELSEIF(USE_CUDA)
    cuda_add_library(${cuda_target} ${ARGN})
  ELSE()
  ENDIF()
endmacro()


##############################################################################
# Add ATen compile options.
# Usage:
#   aten_compile_options(lib_name)
function(aten_compile_options libname)
  target_compile_options(${libname}
    PRIVATE
    -Wall
    -Wextra
    -fexceptions
    -Wno-missing-field-initializers
    -Wno-type-limits
    -Wno-unused-parameter
    -Wno-unknown-warning-option
    -Wno-unknown-pragmas)
  if ($ENV{WERROR})
    target_compile_options(${libname} PRIVATE -Werror)
  endif()
endfunction()


##############################################################################
# Set ATen target properties.
# Usage:
#   aten_set_target_props(lib_name)
function(aten_set_target_props libname)
  if(MSVC AND AT_MKL_MT)
    set_target_properties(${libname} PROPERTIES LINK_FLAGS_RELEASE "/NODEFAULTLIB:${VCOMP_LIB}")
    set_target_properties(${libname} PROPERTIES LINK_FLAGS_DEBUG "/NODEFAULTLIB:${VCOMP_LIB}")
    set_target_properties(${libname} PROPERTIES STATIC_LIBRARY_FLAGS "/NODEFAULTLIB:${VCOMP_LIB}")
  endif()
endfunction()
