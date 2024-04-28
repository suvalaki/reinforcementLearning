cmake_minimum_required(VERSION 3.10)


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

# When this policy is set to NEW, you can call target_link_libraries() 
# on a target from any directory, not just the one where the target was 
# created.
cmake_policy(SET CMP0079 NEW)

# set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OSX_ARCHITECTURES arm64)

# set the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# set the compilers
set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc/13.1.0/bin/gcc-13)
set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/13.1.0/bin/g++-13)

# set additional compile flags
set(GCC_COMPILE_FLAGS "-fdiagnostics-show-template-tree")
add_compile_options(${GCC_COMPILE_FLAGS})


project(reinforce VERSION 0.1)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include(FetchContent)


FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG        0.7.7
)

FetchContent_MakeAvailable(xtl)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG        0.25.0
)

FetchContent_MakeAvailable(xtensor)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
)

FetchContent_MakeAvailable(Catch2)



# set verbose makefile
set(CMAKE_VERBOSE_MAKEFILE on)

include(CTest)
include(Catch)

# Declare the interface library target.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
get_property(PROJECT_INCLUDE_DIRS TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

set_target_properties(${PROJECT_NAME} PROPERTIES
  LINKER_LANGUAGE CXX
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER src/environment.hpp
)

# Add executables for implemented examples
add_subdirectory(examples/bandit)
add_subdirectory(examples/blackjack)


# Add testcase executable
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "test/*.cpp")
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE ${Boost_LIBRARY_DIRS} Catch2::Catch2WithMain xtensor ${PROJECT_NAME})

message(STATUS "Project include dirs: ${PROJECT_INCLUDE_DIRS}")

target_include_directories(tests PUBLIC ${Boost_INCLUDE_DIR} ${Catch2_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIRS} test/)

# Enable testing
catch_discover_tests(tests)