cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  saucer
  GIT_REPOSITORY https://github.com/saucer/saucer
  GIT_TAG        v2.3.0
)

FetchContent_MakeAvailable(saucer)

find_package(Boost 1.65.1 REQUIRED COMPONENTS system)

set(SOURCES
    server.cpp
    game.cpp
    bindings.cpp
    # Add other source files here
)

add_executable(MyProject ${SOURCES})

target_link_libraries(MyProject Boost::system saucer::saucer)