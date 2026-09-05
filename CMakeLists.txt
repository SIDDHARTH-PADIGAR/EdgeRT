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
option(EDGERT_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(EDGERT_BUILD_TESTS "Build unit tests" ON)
option(EDGERT_BUILD_BENCHMARKS "Build benchmark executables" ON)

if(EDGERT_ENABLE_TSAN AND (EDGERT_ENABLE_ASAN OR EDGERT_ENABLE_UBSAN))
    message(FATAL_ERROR "EDGERT_ENABLE_TSAN cannot be combined with ASan/UBSan in the same binary")
endif()

find_package(Threads REQUIRED)

add_library(edgert_core STATIC
    src/tensor.cpp
    src/operator_registry.cpp
    src/builtin_ops.cpp
    src/ops/add.cpp
    src/ops/relu.cpp
    src/ops/matmul.cpp
    src/ops/linear.cpp
    src/graph.cpp
    src/thread_pool.cpp
)
target_link_libraries(edgert_core PUBLIC Threads::Threads)
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

if(EDGERT_ENABLE_TSAN)
    target_compile_options(edgert_core PUBLIC -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(edgert_core PUBLIC -fsanitize=thread)
endif()

add_executable(edgert_demo apps/main.cpp)
target_link_libraries(edgert_demo PRIVATE edgert_core)

if(EDGERT_BUILD_BENCHMARKS)
    add_executable(edgert_bench
        bench/bench_utils.cpp
        bench/bench_ops.cpp
    )
    target_link_libraries(edgert_bench PRIVATE edgert_core)
    # Benchmarks must run optimized even in a Debug configuration, or
    # we'd be measuring unoptimized debug code by accident.
    target_compile_options(edgert_bench PRIVATE -O3)
endif()

if(EDGERT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()