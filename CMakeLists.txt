cmake_minimum_required(VERSION 3.28)
project(alpha_research_2026 VERSION 1.0.0 LANGUAGES CXX)

# ── C++26 standard ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build flags ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address")

# ── Dependencies ──────────────────────────────────────────────────────────────

# Eigen3 (header-only)
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
  # Create alias so find_package-style usage works below
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
  endif()
endif()

# Google Test (for C++ unit tests)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nanobind (Python ↔ C++ bridge)
find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nanobind)
nanobind_add_module(alpha_cpp src/cpp/bindings.cpp)

# ── alpha_cpp_lib: header-only interface library ──────────────────────────────
add_library(alpha_cpp_lib INTERFACE)
target_include_directories(alpha_cpp_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cpp>
  $<INSTALL_INTERFACE:include/alpha_research>
)
target_link_libraries(alpha_cpp_lib INTERFACE Eigen3::Eigen)
target_compile_features(alpha_cpp_lib INTERFACE cxx_std_26)
target_compile_options(alpha_cpp_lib INTERFACE
  $<$<CONFIG:Release>:-O3;-march=native;-mavx2;-mfma;-ffast-math>
)
add_library(alpha_research::alpha_cpp_lib ALIAS alpha_cpp_lib)

# nanobind extension links against our lib
target_link_libraries(alpha_cpp PRIVATE alpha_cpp_lib)

# ── signal_demo binary ────────────────────────────────────────────────────────
add_executable(signal_demo src/cpp/signal_demo.cpp)
target_link_libraries(signal_demo PRIVATE alpha_cpp_lib)

# ── C++ tests ─────────────────────────────────────────────────────────────────
enable_testing()
add_executable(test_engines tests/cpp/test_engines.cpp)
target_link_libraries(test_engines PRIVATE alpha_cpp_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_engines)

# ── Install rules (so `import alpha_cpp` works without absolute paths) ─────────
include(GNUInstallDirs)

# Install header-only lib headers
install(
  DIRECTORY src/cpp/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alpha_research
  FILES_MATCHING PATTERN "*.hpp"
)

# Install the nanobind .so into Python site-packages
install(
  TARGETS alpha_cpp
  LIBRARY DESTINATION ${Python_SITEARCH}
)

# Install CMake package config so other projects can find_package(alpha_research)
install(
  TARGETS alpha_cpp_lib
  EXPORT alpha_research-targets
)
install(
  EXPORT alpha_research-targets
  NAMESPACE alpha_research::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpha_research
)
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/alpha_research-config.cmake.in"
  "${CMAKE_BINARY_DIR}/alpha_research-config.cmake"
  @ONLY
)
install(
  FILES "${CMAKE_BINARY_DIR}/alpha_research-config.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpha_research
)
