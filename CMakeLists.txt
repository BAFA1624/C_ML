cmake_minimum_required(VERSION 3.26)

project(C_ML)

set(CMAKE_CXX_COMPILER g++-13)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wconversion -pedantic -O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -Wno-error=class-memaccess -Wall -Wextra -pedantic -Wconversion -Werror -O3")

add_subdirectory(src)
include_directories(./include)
include_directories(~/include)

add_executable(main src/main.cpp)
target_link_libraries(main d_classic)

add_custom_command(
	TARGET main POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
	COMMENT "Transferring compile_commands.json"
)
