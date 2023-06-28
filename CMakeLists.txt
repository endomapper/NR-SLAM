cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME NR-SLAM)

set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(${PROJECT_NAME})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Compilation flags:
#   -Flag for code optimizations: -O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/g2o)
add_subdirectory(modules)
add_subdirectory(apps)
