cmake_minimum_required(VERSION 3.20)
project(llm-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

include_directories(include)

# ── Third-party: numpy-cpp (header-only, C++20) ──────────────────────────
option(USE_NUMPY_CPP "Use numpy-cpp as accelerated backend for Tensor ops" ON)
if(USE_NUMPY_CPP)
    # Local checkout at /home/sergio/Project/numpy-cpp (preferred)
    if(EXISTS "/home/sergio/Project/numpy-cpp/CMakeLists.txt")
        add_subdirectory("/home/sergio/Project/numpy-cpp" "${CMAKE_BINARY_DIR}/_deps/numpy-cpp")
        message(STATUS "Using local numpy-cpp at /home/sergio/Project/numpy-cpp")
    # Propagate threading if desired
    # set(NP_USE_THREADING ON CACHE BOOL "" FORCE) # uncomment for threading
    else()
        include(FetchContent)
        FetchContent_Declare(numpy-cpp
            GIT_REPOSITORY https://github.com/sergiorandria/numpy-cpp.git
            GIT_TAG dev)
        FetchContent_MakeAvailable(numpy-cpp)
        message(STATUS "Fetched numpy-cpp via FetchContent")
    endif()
endif()

file(GLOB SOURCES "src/*.cpp")
# Note: new modules auto-included via glob
# Exclude examples from main target (they are separate)
list(FILTER SOURCES EXCLUDE REGEX ".*examples/.*")

add_executable(llm-cpp ${SOURCES})

target_include_directories(llm-cpp PRIVATE include)

# Link numpy-cpp if available
if(USE_NUMPY_CPP AND TARGET numpy-cpp)
    target_link_libraries(llm-cpp PRIVATE numpy-cpp)
    target_compile_definitions(llm-cpp PRIVATE USE_NUMPY_CPP=1)
    message(STATUS "llm-cpp: USE_NUMPY_CPP enabled (C++20, SIMD, linalg)")
elseif(USE_NUMPY_CPP AND TARGET numpy-cpp::numpy-cpp)
    target_link_libraries(llm-cpp PRIVATE numpy-cpp::numpy-cpp)
    target_compile_definitions(llm-cpp PRIVATE USE_NUMPY_CPP=1)
    message(STATUS "llm-cpp: USE_NUMPY_CPP enabled via alias")
endif()

# Optional: OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llm-cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

option(BUILD_TESTS "Build tests" ON)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

if(ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

install(TARGETS llm-cpp DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

enable_testing()
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Benchmark target (uses numpy if enabled)
add_executable(bench_matmul scripts/bench_matmul.cpp src/tensor.cpp)
if(USE_NUMPY_CPP AND TARGET numpy-cpp)
  target_link_libraries(bench_matmul PRIVATE numpy-cpp)
  target_compile_definitions(bench_matmul PRIVATE USE_NUMPY_CPP=1)
endif()
