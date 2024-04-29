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
find_package(SQLite3)

FetchContent_Declare(
  saucer
  GIT_REPOSITORY https://github.com/saucer/saucer
  GIT_TAG        v2.1.0
)
FetchContent_MakeAvailable(saucer)

FetchContent_Declare(
  zxorm
  GIT_REPOSITORY https://github.com/crabmandable/zxorm.git
  GIT_TAG 0.1.3
)
FetchContent_MakeAvailable(zxorm)

set(SOURCES
    src/main.cpp
    src/server.cpp
    src/game.cpp
    src/bindings.cpp
    src/tracing.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${SQLite3_INCLUDE_DIRS} ${zxorm_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::system saucer::saucer ${SQLite3_LIBRARIES})