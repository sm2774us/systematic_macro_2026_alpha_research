cmake_minimum_required(VERSION 3.28)
project(alpha_research_2026 VERSION 1.0.0 LANGUAGES CXX)

# ── C++23 standard ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
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

# ── alpha_cpp_lib: STATIC library for RTTI Stability ───────────────────────
# Changed from INTERFACE to STATIC to serve as the "Strong Anchor" 
add_library(alpha_cpp_lib STATIC 
    src/cpp/pdrrm_engine.cpp
    src/cpp/strategies_engine.cpp
    src/cpp/portfolio_optimizer.cpp
)

# Mandatory for linking into a shared object (.so) 
set_target_properties(alpha_cpp_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Fix: Use BUILD_INTERFACE to resolve the export path error 
target_include_directories(alpha_cpp_lib 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cpp>
        $<INSTALL_INTERFACE:include/alpha_research>
)

# FIX: Changed from INTERFACE to PUBLIC so STATIC lib can find Eigen during its own build [cite: 13, 18]
target_link_libraries(alpha_cpp_lib PUBLIC Eigen3::Eigen)
target_compile_features(alpha_cpp_lib PUBLIC cxx_std_23)

# RTTI and Visibility flags for stability against std::bad_cast [cite: 19]
target_compile_options(alpha_cpp_lib PRIVATE -frtti)
set_target_properties(alpha_cpp_lib PROPERTIES 
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

# ── nanobind module ──────────────────────────────────────────────────────────
nanobind_add_module(alpha_cpp src/cpp/bindings.cpp)

# Stabilize registry and registry symbols [cite: 15, 19]
target_compile_definitions(alpha_cpp PRIVATE NB_DOMAIN=alpha_research_production)
set_target_properties(alpha_cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

if(UNIX AND NOT APPLE)
    target_link_options(alpha_cpp PRIVATE "-Wl,--no-as-needed")
endif()

# Pulls in compiled objects from alpha_cpp_lib [cite: 20]
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

# ── Install rules ─────────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(
  DIRECTORY src/cpp/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/alpha_research
  FILES_MATCHING PATTERN "*.hpp"
)

install(
  TARGETS alpha_cpp
  LIBRARY DESTINATION ${Python_SITEARCH}
)

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
