cmake_minimum_required(VERSION 3.10)
project(Alpha VERSION 1.0)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.c)

add_library(alpha ${SOURCES})

target_include_directories(alpha PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(alpha PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OpenMP_C_FOUND)
	target_link_libraries(alpha PUBLIC OpenMP::OpenMP_C)
endif()

if(NOT WIN32)
	target_link_libraries(alpha m)
endif()