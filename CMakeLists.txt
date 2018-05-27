cmake_minimum_required(VERSION 3.2)

project (MTCNN C CXX)

set_property (GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory (lib)
add_subdirectory (sample)

enable_testing()