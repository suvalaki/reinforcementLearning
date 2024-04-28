cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc/13.1.0/bin/gcc-13)
set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/13.1.0/bin/g++-13)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(Qt5_DIR  /opt/homebrew/opt/qt@5/lib/cmake/Qt5 )

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OSX Architectures: ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")

project(TicTacToe)
include(FetchContent)
find_package(Boost 1.65.1 REQUIRED COMPONENTS system)
find_package(Qt5 COMPONENTS Core)

FetchContent_Declare(
  saucer
  GIT_REPOSITORY https://github.com/saucer/saucer
  GIT_TAG        v2.1.0
)

FetchContent_MakeAvailable(saucer)

set(SOURCES
    src/main.cpp
    src/server.cpp
    src/game.cpp
    src/bindings.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::system saucer::saucer)