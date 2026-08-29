include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(edgert_tests
    test_tensor.cpp
)
target_link_libraries(edgert_tests PRIVATE edgert_core gtest_main)

include(GoogleTest)
gtest_discover_tests(edgert_tests)