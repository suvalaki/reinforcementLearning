cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc/13.1.0/bin/gcc-13)
set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/13.1.0/bin/g++-13)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_CXX_STANDARD 20)
#set(Qt6_DIR /opt/homebrew/opt/qt/lib/cmake/Qt6)
set(Qt5_DIR  /opt/homebrew/opt/qt@5/lib/cmake/Qt5 )

set(CMAKE_OSX_ARCHITECTURES arm64)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)


message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OSX Architectures: ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Qt6_DIR: ${Qt6_DIR}")


project(MyProject)
include(FetchContent)

#find_package(Qt6 COMPONENTS Core)
find_package(Qt5 COMPONENTS Core)

FetchContent_Declare(
  saucer
  GIT_REPOSITORY https://github.com/saucer/saucer
  #GIT_TAG        v2.2.0
  GIT_TAG        v2.1.0
  #CMAKE_ARGS -DCMAKE_OSX_ARCHITECTURES=arm64
)

FetchContent_MakeAvailable(saucer)

#find_package(Boost 1.65.1 REQUIRED COMPONENTS system)

set(SOURCES
    #test.cpp
    main.cpp
    #server.cpp
    #game.cpp
    #bindings.cpp
    # Add other source files here
)

add_executable(MyProject ${SOURCES})

# Boost::system

target_link_libraries(MyProject PRIVATE saucer::saucer)