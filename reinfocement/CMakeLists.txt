cmake_minimum_required(VERSION 3.10)

set(CATCH_INSTALL_EXTRAS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)

# set the project name
project(ReinforcementLearning VERSION 0.1)


# Declare the library target.
add_library(${PROJECT_NAME} SHARED
    src/environment.hpp
)



# Configure the directories to search for header files.
target_include_directories(${PROJECT_NAME} PUBLIC src src/bandit src/policy)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
# Set the version property.
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
# Set the shared object version property to the project's major version.
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
# Set the public header property to the one with the actual API.
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER 
    src/environment.hpp
)

file(GLOB_RECURSE bandits CONFIGURE_DEPENDS   "src/*.hpp" "src/*.cpp" "src/bandit/*.hpp" "src/policy/*.hpp")
add_executable(MultiArmedBandit "src/bandit/run_bandit.cpp")
target_include_directories(MultiArmedBandit PUBLIC src )
set(GCC_COVERAGE_COMPILE_FLAGS "-fconcepts-diagnostics-depth=2")



# Add testcase executable
enable_testing()
find_package(Catch2 3 REQUIRED) 


file(GLOB_RECURSE testfiles CONFIGURE_DEPENDS "test/*.h" "test/*.cpp")
add_executable(tests ${testfiles})

# add_subdirectory(lib/Catch2)
target_link_libraries(tests INTERFACE Catch2::Catch2)
# set_target_properties(tests PROPERTIES
#              CXX_STANDARD 17
#              CXX_EXTENSIONS OFF
#              )

target_include_directories(tests PUBLIC src test)

# include(CTest)
# include(lib/Catch2/extras/Catch.cmake)
# include(lib/Catch2/extras/ParseAndAddCatchTests.cmake)
# ParseAndAddCatchTests(tests)

# # Add the tests to be run 
# # add_test(allTests tests)

# # Run tests after each build 
# # add_custom_command(TARGET tests 
# #                   POST_BUILD
# #                   COMMAND ctest -C $<CONFIGURATION> --output-on-failure)
# # Instead just run ctest explicitly