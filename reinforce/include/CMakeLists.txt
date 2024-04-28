find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(Catch2 3 REQUIRED HINTS "/opt/homebrew/Cellar/catch2/3.3.2_1")

# Declare the interface library target.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(${PROJECT_NAME} PROPERTIES
  LINKER_LANGUAGE CXX
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER src/environment.hpp
)
