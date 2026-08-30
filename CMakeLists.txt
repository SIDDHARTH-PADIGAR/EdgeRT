cmake_minimum_required(VERSION 3.16)
project(EdgeRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(EDGERT_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(EDGERT_ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(EDGERT_BUILD_TESTS "Build unit tests" ON)

add_library(edgert_core STATIC
    src/tensor.cpp
    src/operator_registry.cpp
    src/builtin_ops.cpp
    src/ops/add.cpp
    src/ops/relu.cpp
    src/ops/matmul.cpp
    src/graph.cpp
)
target_include_directories(edgert_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(edgert_core PRIVATE -Wall -Wextra -Wpedantic)

if(EDGERT_ENABLE_ASAN)
    target_compile_options(edgert_core PUBLIC -fsanitize=address -fno-omit-frame-pointer -g)
    target_link_options(edgert_core PUBLIC -fsanitize=address)
endif()

if(EDGERT_ENABLE_UBSAN)
    target_compile_options(edgert_core PUBLIC -fsanitize=undefined -fno-omit-frame-pointer -g)
    target_link_options(edgert_core PUBLIC -fsanitize=undefined)
endif()

add_executable(edgert_demo apps/main.cpp)
target_link_libraries(edgert_demo PRIVATE edgert_core)

if(EDGERT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()