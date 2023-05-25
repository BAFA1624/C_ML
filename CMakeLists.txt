cmake_minimum_required(VERSION 3.26)

project(C_ML)

set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -pedantic -Werror -O3")

add_subdirectory(src)
include_directories(./include)

add_executable(main src/main.cpp)