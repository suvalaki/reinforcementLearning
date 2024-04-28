add_executable(blackjack src/main.cpp)
target_link_libraries(bandits reinforce)
target_include_directories(blackjack PUBLIC include/ ${PROJECT_INCLUDE_DIRS})

# Testing
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "test/*.cpp")
add_executable(blackjack_tests ${TEST_SOURCES})
target_link_libraries(blackjack_tests PRIVATE reinforce Catch2::Catch2WithMain ${PROJECT_NAME} xtensor ${Boost_LIBRARY_DIRS})
target_include_directories(blackjack_tests PUBLIC include/ ${PROJECT_INCLUDE_DIRS} ${Catch2_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
catch_discover_tests(blackjack_tests)
