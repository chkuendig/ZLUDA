cmake_minimum_required(VERSION 3.18)
project(physx5_d3d11_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- PhysX 5 SDK ---
set(PHYSX_ROOT "${CMAKE_SOURCE_DIR}/../physx5_gpu/PhysX/physx" CACHE PATH
    "Path to PhysX 5 SDK root (physx/ subdirectory)")

# PhysX include directories
set(PHYSX_INCLUDE_DIRS
    "${PHYSX_ROOT}/include"
    "${PHYSX_ROOT}/../pxshared/include"
)

# PhysX library directory (adjust for build type)
set(PHYSX_BUILD_TYPE "release" CACHE STRING "PhysX build type (release/debug/checked)")
if(WIN32)
    set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/win.x86_64.vc143.mt/${PHYSX_BUILD_TYPE}")
else()
    set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/linux.x86_64/${PHYSX_BUILD_TYPE}")
endif()

# PhysX libraries (static linking)
set(PHYSX_LIBRARIES
    PhysX_static_64
    PhysXExtensions_static_64
    PhysXCommon_static_64
    PhysXFoundation_static_64
    PhysXPvdSDK_static_64
)

# --- Build the test executable ---
add_executable(test_physx_d3d11
    src/main.cpp
    src/physx_scene.cpp
    src/d3d11_renderer.cpp
    src/cuda_d3d11_interop.cpp
)

target_include_directories(test_physx_d3d11 PRIVATE
    ${PHYSX_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/src"
)

# PhysX linking
target_link_directories(test_physx_d3d11 PRIVATE ${PHYSX_LIB_DIR})

if(WIN32)
    # Windows: link D3D11 and PhysX
    target_link_libraries(test_physx_d3d11
        ${PHYSX_LIBRARIES}
        d3d11 dxgi d3dcompiler
    )
    # PhysX GPU on Windows needs the GPU DLL at runtime
    target_compile_definitions(test_physx_d3d11 PRIVATE
        PX_PHYSX_STATIC_LIB
        _WIN32
    )
else()
    # Linux: PhysX only (no D3D11) — D3D11 tests only run under Wine
    target_link_libraries(test_physx_d3d11
        ${PHYSX_LIBRARIES}
        pthread dl
    )
    target_compile_definitions(test_physx_d3d11 PRIVATE
        PX_PHYSX_STATIC_LIB
    )
endif()

# --- MinGW cross-compilation support ---
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=../mingw-w64.cmake ..
if(MINGW)
    target_link_libraries(test_physx_d3d11
        ${PHYSX_LIBRARIES}
        d3d11 dxgi d3dcompiler
    )
    target_compile_definitions(test_physx_d3d11 PRIVATE
        PX_PHYSX_STATIC_LIB
        _WIN32
    )
endif()
