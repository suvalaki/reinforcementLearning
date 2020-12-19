cmake_minimum_required(VERSION 3.10)

set(CATCH_INSTALL_EXTRAS ON)
set(CMAKE_BUILD_TYPE Debug)

# set the project name
project(MultiArmedBandit)

# add the executable
add_executable(MultiArmedBandit src/bandit_run.cpp)
set_target_properties(MultiArmedBandit PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            )
target_include_directories(MultiArmedBandit PUBLIC src)


# Add testcase executable
enable_testing()
file(GLOB_RECURSE testfiles CONFIGURE_DEPENDS "test/*.h" "test/*.cpp")
add_executable(tests ${testfiles})
add_subdirectory(lib/Catch2)
target_link_libraries(tests INTERFACE Catch2::Catch2)
set_target_properties(tests PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            )

target_include_directories(tests PUBLIC src test)

include(CTest)
include(lib/Catch2/extras/Catch.cmake)
include(lib/Catch2/extras/ParseAndAddCatchTests.cmake)
ParseAndAddCatchTests(tests)

# Add the tests to be run 
# add_test(allTests tests)

# Run tests after each build 
# add_custom_command(TARGET tests 
#                   POST_BUILD
#                   COMMAND ctest -C $<CONFIGURATION> --output-on-failure)
# Instead just run ctest explicitly