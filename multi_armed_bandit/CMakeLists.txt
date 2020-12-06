cmake_minimum_required(VERSION 3.10)

# set the project name
project(MultiArmedBandit)

# add the executable
add_executable(MultiArmedBandit src/bandit_run.cpp)
set_target_properties(MultiArmedBandit PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            )
include_directories(src)

# Add testcase executable
add_executable(tests test/main.cpp)
set_target_properties(tests PROPERTIES
            CXX_STANDARD 17
            CXX_EXTENSIONS OFF
            )

include_directories(test)

# Include catch2 from subdirectory
add_subdirectory(lib/Catch2)
target_link_libraries(tests Catch2::Catch2)