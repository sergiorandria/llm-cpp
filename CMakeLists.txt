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
# Portable: -march=native is opt-in via ENABLE_NATIVE (off by default for reproducibility)
option(ENABLE_NATIVE "Enable -march=native for Release" OFF)
if(ENABLE_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

include_directories(include)

# ── Third-party: numpy-cpp (header-only, C++20) ──────────────────────────
option(USE_NUMPY_CPP "Use numpy-cpp as accelerated backend for Tensor ops" ON)
option(NUMPY_CPP_LOCAL_PATH "Local path for numpy-cpp (empty = FetchContent)" "/home/sergio/Project/numpy-cpp")
if(USE_NUMPY_CPP)
    if(NUMPY_CPP_LOCAL_PATH AND EXISTS "${NUMPY_CPP_LOCAL_PATH}/CMakeLists.txt")
        add_subdirectory("${NUMPY_CPP_LOCAL_PATH}" "${CMAKE_BINARY_DIR}/_deps/numpy-cpp")
        message(STATUS "Using local numpy-cpp at ${NUMPY_CPP_LOCAL_PATH}")
    else()
        include(FetchContent)
        FetchContent_Declare(numpy-cpp
            GIT_REPOSITORY https://github.com/sergiorandria/numpy-cpp.git
            GIT_TAG main)
        FetchContent_MakeAvailable(numpy-cpp)
        message(STATUS "Fetched numpy-cpp via FetchContent (main)")
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

# Optional: OpenBLAS/MKL
option(USE_OPENBLAS "Use OpenBLAS cblas_sgemm for matmul" OFF)
if(USE_OPENBLAS)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_include_directories(llm-cpp PRIVATE ${OpenBLAS_INCLUDE_DIRS})
        target_link_libraries(llm-cpp PRIVATE ${OpenBLAS_LIBRARIES})
        target_compile_definitions(llm-cpp PRIVATE USE_OPENBLAS=1)
        message(STATUS "llm-cpp: USE_OPENBLAS enabled — cblas_sgemm")
    else()
        message(WARNING "USE_OPENBLAS=ON but OpenBLAS not found — fallback to blocked GEMM")
    endif()
endif()

# I87: CUDA backend (optional; CPU-only build unchanged when absent)
option(USE_CUDA "Use CUDA backend for matmul (requires CUDAToolkit)" OFF)
if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(llm-cpp PRIVATE CUDA::cudart CUDA::cublas)
        target_compile_definitions(llm-cpp PRIVATE USE_CUDA=1)
        message(STATUS "llm-cpp: USE_CUDA enabled — cublas")
    else()
        message(WARNING "USE_CUDA=ON but CUDAToolkit not found — staying on CPU")
    endif()
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

# G63: shared C ABI library for Python ctypes (no pybind11 needed)
add_library(llm-c-api SHARED
  src/c_api.cpp src/model.cpp src/attention.cpp src/tensor.cpp src/transformer.cpp
  src/tokenizer.cpp src/sampling.cpp src/kv_cache.cpp src/flash_attention.cpp
  src/fused.cpp src/layernorm.cpp src/utils.cpp src/config.cpp src/profiling.cpp)
target_include_directories(llm-c-api PRIVATE include)
set_target_properties(llm-c-api PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if(USE_NUMPY_CPP AND TARGET numpy-cpp)
  target_link_libraries(llm-c-api PRIVATE numpy-cpp)
  target_compile_definitions(llm-c-api PRIVATE USE_NUMPY_CPP=1)
elseif(USE_NUMPY_CPP AND TARGET numpy-cpp::numpy-cpp)
  target_link_libraries(llm-c-api PRIVATE numpy-cpp::numpy-cpp)
  target_compile_definitions(llm-c-api PRIVATE USE_NUMPY_CPP=1)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(llm-c-api PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS llm-c-api DESTINATION lib)

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

# I89: inference benchmark (tokens/sec, p50/p95)
add_executable(bench_infer scripts/bench_infer.cpp src/model.cpp src/attention.cpp src/tensor.cpp
  src/transformer.cpp src/tokenizer.cpp src/sampling.cpp src/kv_cache.cpp src/flash_attention.cpp
  src/fused.cpp src/layernorm.cpp src/utils.cpp src/config.cpp src/profiling.cpp src/pool.cpp)
target_include_directories(bench_infer PRIVATE include)
if(USE_NUMPY_CPP AND TARGET numpy-cpp)
  target_link_libraries(bench_infer PRIVATE numpy-cpp)
  target_compile_definitions(bench_infer PRIVATE USE_NUMPY_CPP=1)
elseif(USE_NUMPY_CPP AND TARGET numpy-cpp::numpy-cpp)
  target_link_libraries(bench_infer PRIVATE numpy-cpp::numpy-cpp)
  target_compile_definitions(bench_infer PRIVATE USE_NUMPY_CPP=1)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(bench_infer PRIVATE OpenMP::OpenMP_CXX)
endif()
