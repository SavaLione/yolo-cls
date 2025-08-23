#######################################################################
# Copyright (C) 2025 Savelii Pototskii (savalione.com)
# 
# Author: Savelii Pototskii <savelii.pototskii@gmail.com>
# 
# This file is part of yolo-cls.
# 
# yolo-cls is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
# 
# yolo-cls is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with yolo-cls. If not, see <https://www.gnu.org/licenses/>.
#######################################################################
cmake_minimum_required (VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(yolo-cls
    VERSION "1.0.0"
    DESCRIPTION "yolo-cls is command-line tool for YOLO-based image classification."
    HOMEPAGE_URL "https://github.com/SavaLione/yolo-cls"
    LANGUAGES C CXX
)

set(PROJECT_AUTHOR "Savelii Pototskii (savalione.com)")
set(PROJECT_LICENSE "GPL-3.0-or-later")

# Project Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF)

# The project options
option(YOLOCLS_USE_CUDA "Use Nvidia CUDA backend" OFF)

# Provide compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set a default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# Dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc dnn)
find_package(ONNXRuntime REQUIRED)

# Sources and executable definition
set(YOLOCLS_SRC
    src/yolo-cls.cpp
    src/yolo.cpp
    src/utils.cpp
    src/tsqueue.cpp
    src/xgetopt/xgetopt.c
)

add_executable(${PROJECT_NAME} ${YOLOCLS_SRC})

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    ONNXRuntime::ONNXRuntime
)

# Configuration file generation
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h"
)

# Link everything statically
target_link_options(${PROJECT_NAME} PRIVATE
    -static-libstdc++
    -static-libgcc
)

# GNUInstallDirs to use standard directory variables
include(GNUInstallDirs)

# Set the RPATH for the installed executable
# set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

# Ensure the linker uses this RPATH for the installed target
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# The installation rule for the main executable
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install ONNX Runtime library
get_target_property(ONNXRUNTIME_LIB_LOCATION ONNXRuntime::ONNXRuntime IMPORTED_LOCATION)
get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB_LOCATION} DIRECTORY)

install(DIRECTORY ${ONNXRUNTIME_LIB_DIR}/ DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILES_MATCHING
        PATTERN "libonnxruntime.so*"
        PATTERN "libonnxruntime_providers_shared.so*"
        PATTERN "onnxruntime.dll*"
        PATTERN "onnxruntime_providers_shared.dll*"
        PATTERN "cmake" EXCLUDE
        PATTERN "pkgconfig" EXCLUDE
)

# Install OpenCV libraries
foreach(CV_LIB ${OpenCV_LIBS})
    get_target_property(LIB_PATH ${CV_LIB} IMPORTED_LOCATION_RELEASE)
    install(FILES ${LIB_PATH} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endforeach()
