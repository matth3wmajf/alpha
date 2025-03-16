cmake_minimum_required(VERSION 3.10)
project(Alpha VERSION 1.0)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.c)

add_library(alpha ${SOURCES})

target_include_directories(alpha PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(alpha PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)