# For example, in bandits/CMakeLists.txt
add_executable(bandits src/main.cpp)
target_link_libraries(bandits reinforce xtensor ${Boost_LIBRARY_DIRS})
target_include_directories(bandits PRIVATE include/ ${PROJECT_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})


# Testing
# file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "test/*.cpp")
# add_executable(bandit_tests ${TEST_SOURCES})
# target_link_libraries(bandit_tests PRIVATE Catch2::Catch2WithMain ${PROJECT_NAME} xtensor)
# target_include_directories(bandit_tests PUBLIC src ${Catch2_INCLUDE_DIRS})
# catch_discover_tests(bandit_tests)
