project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(testable_sources src/ukf.cpp src/tools.cpp)
set(sources src/main.cpp ${testable_sources})


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 


add_executable(UnscentedKF ${sources})

target_link_libraries(UnscentedKF z ssl uv uWS)

# adding test executable

enable_testing()
add_subdirectory(test)
include_directories(${gtest_SOURCE_DIR}/include)
include_directories(src)

file(GLOB SRC_TEST ${PROJECT_SOURCE_DIR}/test/*.cpp)
add_executable(ukf-test ${SRC_TEST} ${testable_sources})
target_link_libraries(ukf-test gtest_main z ssl uv uWS)
add_test(NAME gtest-run COMMAND ukf-test)