cmake_minimum_required(VERSION 3.16)
project(llm-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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

file(GLOB SOURCES "src/*.cpp")

add_executable(llm-cpp ${SOURCES})

target_include_directories(llm-cpp PRIVATE include)

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
