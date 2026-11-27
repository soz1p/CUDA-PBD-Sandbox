cmake_minimum_required(VERSION 3.18)
project(CudaPbdSandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# CUDA is optional: without it the CPU solvers, the benchmark and the viewer still build.
option(PBD_ENABLE_CUDA "Build the GPU solvers when a CUDA compiler is found" ON)
set(PBD_WITH_CUDA OFF)
if(PBD_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75 86 89)   # RTX 20 / 30 / 40 series
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(PBD_WITH_CUDA ON)
  else()
    message(STATUS "CUDA compiler not found: building CPU solvers only")
  endif()
endif()

add_library(pbd_core
  src/cloth.cpp
  src/blocks.cpp
  src/cpu_solver.cpp
  src/cpu_active_solver.cpp)
target_include_directories(pbd_core PUBLIC include)

if(PBD_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(pbd_core PRIVATE src/gpu_solver.cu)
  target_compile_options(pbd_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>")
  target_compile_definitions(pbd_core PUBLIC PBD_WITH_CUDA)
  target_link_libraries(pbd_core PUBLIC CUDA::cudart)
endif()

if(MSVC)
  target_compile_options(pbd_core PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
  target_compile_options(pbd_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
else()
  target_compile_options(pbd_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

add_executable(pbd_bench apps/bench.cpp)
target_link_libraries(pbd_bench PRIVATE pbd_core)

if(WIN32)
  # Real-time viewer: Win32 + OpenGL 3.3, no third-party libraries.
  add_executable(pbd_viewer WIN32 apps/viewer/main.cpp apps/viewer/gl_api.cpp apps/viewer/cloth_renderer.cpp)
  target_include_directories(pbd_viewer PRIVATE apps/viewer)
  target_link_libraries(pbd_viewer PRIVATE pbd_core opengl32 gdi32 user32)
endif()

if(WIN32)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT pbd_viewer)
else()
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT pbd_bench)
endif()
