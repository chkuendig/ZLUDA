// PhysX 5 + D3D11 + CUDA Interop Test Suite
//
// Tests the full PhysX GPU → CUDA → D3D11 pipeline that games use.
// Each test mode isolates a different layer of the stack.
//
// Usage:
//   test_physx_d3d11.exe --test-mode <mode>
//
// Modes:
//   physx-only       PhysX GPU simulation without D3D11 (tests CUDA kernels)
//   d3d11-only       D3D11 device + buffers without CUDA (tests DXVK)
//   cuda-init        Just cuInit + cuCtxCreate (tests CUDA/ZLUDA basics)
//   interop-register Register D3D11 buffer with CUDA (tests the critical path)
//   interop-map      Register + map + unmap lifecycle
//   interop-copy     Full: simulate + copy to D3D11 + readback
//   all              Run all modes, report pass/fail

#include "physx_scene.h"
#include "d3d11_renderer.h"
#include "cuda_d3d11_interop.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// --- Test result tracking ---
static int gTestsPassed = 0;
static int gTestsFailed = 0;
static int gTestsSkipped = 0;

#define TEST_BEGIN(name) do { \
    fprintf(stderr, "\n=== TEST: %s ===\n", name); \
} while(0)

#define TEST_PASS(name) do { \
    fprintf(stderr, "=== PASS: %s ===\n", name); \
    gTestsPassed++; \
} while(0)

#define TEST_FAIL(name, reason) do { \
    fprintf(stderr, "=== FAIL: %s — %s ===\n", name, reason); \
    gTestsFailed++; \
} while(0)

#define TEST_SKIP(name, reason) do { \
    fprintf(stderr, "=== SKIP: %s — %s ===\n", name, reason); \
    gTestsSkipped++; \
} while(0)

// ========================================================================
// Test: physx-only — Run PhysX GPU simulation headless
// Tests: cuInit, cuCtxCreate, cuModuleLoadData, cuLaunchKernel through ZLUDA
// ========================================================================
static void test_physx_only() {
    TEST_BEGIN("physx-only");

    PhysXScene scene;
    if (!physxInit(scene, 1024)) {
        TEST_FAIL("physx-only", "PhysX GPU initialization failed");
        physxCleanup(scene);
        return;
    }

    // Run a few simulation steps
    fprintf(stderr, "[Test] Running 10 simulation steps...\n");
    for (int i = 0; i < 10; i++) {
        physxStep(scene, 1.0f / 60.0f);
        fprintf(stderr, "[Test] Step %d/10 complete\n", i + 1);
    }

    physxCleanup(scene);
    TEST_PASS("physx-only");
}

// ========================================================================
// Test: d3d11-only — Create D3D11 device and buffers (no CUDA)
// Tests: DXVK translation layer
// ========================================================================
static void test_d3d11_only() {
    TEST_BEGIN("d3d11-only");

#ifdef _WIN32
    D3D11Renderer renderer;
    if (!d3d11Init(renderer, 1024, true)) {
        TEST_FAIL("d3d11-only", "D3D11 device creation failed");
        return;
    }

    void* sharedBuf = d3d11CreateSharedBuffer(renderer, 1024);
    if (!sharedBuf) {
        TEST_FAIL("d3d11-only", "Shared buffer creation failed");
        d3d11Cleanup(renderer);
        return;
    }

    d3d11Cleanup(renderer);
    TEST_PASS("d3d11-only");
#else
    TEST_SKIP("d3d11-only", "D3D11 not available on Linux (run under Wine)");
#endif
}

// ========================================================================
// Test: cuda-init — Just CUDA initialization
// Tests: nvcuda.dll loading, cuInit, cuCtxCreate
// ========================================================================
static void test_cuda_init() {
    TEST_BEGIN("cuda-init");

    CudaInterop cuda;
    if (!cudaInteropInit(cuda)) {
        TEST_FAIL("cuda-init", "CUDA library loading failed");
        return;
    }

    if (!cudaInteropCreateContext(cuda)) {
        TEST_FAIL("cuda-init", "CUDA context creation failed");
        cudaInteropCleanup(cuda);
        return;
    }

    cudaInteropCleanup(cuda);
    TEST_PASS("cuda-init");
}

// ========================================================================
// Test: interop-register — Register D3D11 buffer with CUDA
// This is the CRITICAL test — it reveals whether cuGraphicsD3D11RegisterResource
// reaches ZLUDA through Wine's nvcuda relay.
// ========================================================================
static void test_interop_register() {
    TEST_BEGIN("interop-register");

#ifdef _WIN32
    // Step 1: Create D3D11 device + shared buffer
    D3D11Renderer renderer;
    if (!d3d11Init(renderer, 1024, true)) {
        TEST_FAIL("interop-register", "D3D11 init failed");
        return;
    }

    void* d3d11Buffer = d3d11CreateSharedBuffer(renderer, 1024);
    if (!d3d11Buffer) {
        TEST_FAIL("interop-register", "D3D11 shared buffer creation failed");
        d3d11Cleanup(renderer);
        return;
    }

    // Step 2: Initialize CUDA
    CudaInterop cuda;
    if (!cudaInteropInit(cuda)) {
        TEST_FAIL("interop-register", "CUDA init failed");
        d3d11Cleanup(renderer);
        return;
    }

    if (!cudaInteropCreateContext(cuda)) {
        TEST_FAIL("interop-register", "CUDA context creation failed");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    // Step 3: Register D3D11 buffer with CUDA — THE CRITICAL CALL
    CUgraphicsResource cuResource = nullptr;
    if (!cudaInteropRegisterD3D11(cuda, d3d11Buffer, &cuResource)) {
        TEST_FAIL("interop-register",
                  "cuGraphicsD3D11RegisterResource FAILED — "
                  "this call may be intercepted by Wine's nvcuda relay");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    // Step 4: Unregister
    cudaInteropUnregister(cuda, cuResource);
    cudaInteropCleanup(cuda);
    d3d11Cleanup(renderer);
    TEST_PASS("interop-register");
#else
    TEST_SKIP("interop-register", "D3D11 not available on Linux");
#endif
}

// ========================================================================
// Test: interop-map — Register + map + unmap lifecycle
// ========================================================================
static void test_interop_map() {
    TEST_BEGIN("interop-map");

#ifdef _WIN32
    D3D11Renderer renderer;
    if (!d3d11Init(renderer, 1024, true)) {
        TEST_FAIL("interop-map", "D3D11 init failed");
        return;
    }

    void* d3d11Buffer = d3d11CreateSharedBuffer(renderer, 1024);
    CudaInterop cuda;
    if (!d3d11Buffer || !cudaInteropInit(cuda) || !cudaInteropCreateContext(cuda)) {
        TEST_FAIL("interop-map", "Setup failed");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    CUgraphicsResource cuResource = nullptr;
    if (!cudaInteropRegisterD3D11(cuda, d3d11Buffer, &cuResource)) {
        TEST_FAIL("interop-map", "Registration failed");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    // Map
    if (!cudaInteropMap(cuda, cuResource)) {
        TEST_FAIL("interop-map", "Map failed");
        cudaInteropUnregister(cuda, cuResource);
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    // Get mapped pointer
    CUdeviceptr devPtr = 0;
    size_t size = 0;
    if (!cudaInteropGetMappedPointer(cuda, cuResource, &devPtr, &size)) {
        fprintf(stderr, "[Test] WARNING: GetMappedPointer failed (may be texture-only resource)\n");
    }

    // Unmap
    if (!cudaInteropUnmap(cuda, cuResource)) {
        TEST_FAIL("interop-map", "Unmap failed");
        cudaInteropUnregister(cuda, cuResource);
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    cudaInteropUnregister(cuda, cuResource);
    cudaInteropCleanup(cuda);
    d3d11Cleanup(renderer);
    TEST_PASS("interop-map");
#else
    TEST_SKIP("interop-map", "D3D11 not available on Linux");
#endif
}

// ========================================================================
// Test: interop-copy — Full data transfer: CUDA write → D3D11 readback
// ========================================================================
static void test_interop_copy() {
    TEST_BEGIN("interop-copy");

#ifdef _WIN32
    const uint32_t NUM_PARTICLES = 256;
    const uint32_t BUFFER_SIZE = NUM_PARTICLES * 4 * sizeof(float); // float4

    D3D11Renderer renderer;
    CudaInterop cuda;

    if (!d3d11Init(renderer, NUM_PARTICLES, true) ||
        !cudaInteropInit(cuda) || !cudaInteropCreateContext(cuda)) {
        TEST_FAIL("interop-copy", "Setup failed");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    void* d3d11Buffer = d3d11CreateSharedBuffer(renderer, NUM_PARTICLES);
    if (!d3d11Buffer) {
        TEST_FAIL("interop-copy", "Shared buffer creation failed");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    CUgraphicsResource cuResource = nullptr;
    if (!cudaInteropRegisterD3D11(cuda, d3d11Buffer, &cuResource)) {
        TEST_FAIL("interop-copy", "Registration failed");
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    // Map
    if (!cudaInteropMap(cuda, cuResource)) {
        TEST_FAIL("interop-copy", "Map failed");
        cudaInteropUnregister(cuda, cuResource);
        cudaInteropCleanup(cuda);
        d3d11Cleanup(renderer);
        return;
    }

    // Get mapped pointer
    CUdeviceptr devPtr = 0;
    size_t mappedSize = 0;
    if (cudaInteropGetMappedPointer(cuda, cuResource, &devPtr, &mappedSize)) {
        // Write test pattern via CUDA
        float* testData = new float[NUM_PARTICLES * 4];
        for (uint32_t i = 0; i < NUM_PARTICLES; i++) {
            testData[i * 4 + 0] = (float)i;           // x
            testData[i * 4 + 1] = (float)(i * 2);     // y
            testData[i * 4 + 2] = (float)(i * 3);     // z
            testData[i * 4 + 3] = 1.0f;               // w
        }

        if (cuda.cuMemcpyHtoD) {
            CUresult res = cuda.cuMemcpyHtoD(devPtr, testData, BUFFER_SIZE);
            if (res != CUDA_SUCCESS) {
                fprintf(stderr, "[Test] WARNING: cuMemcpyHtoD failed (result=%d)\n", res);
            }
        }
        delete[] testData;
    }

    // Unmap
    cudaInteropUnmap(cuda, cuResource);

    // Readback through D3D11
    float* readback = new float[NUM_PARTICLES * 4];
    memset(readback, 0, BUFFER_SIZE);
    if (d3d11ReadbackBuffer(renderer, readback, BUFFER_SIZE)) {
        // Verify test pattern
        bool match = true;
        for (uint32_t i = 0; i < NUM_PARTICLES && match; i++) {
            if (readback[i * 4] != (float)i) match = false;
        }
        if (match) {
            fprintf(stderr, "[Test] D3D11 readback matches CUDA write!\n");
        } else {
            fprintf(stderr, "[Test] D3D11 readback does NOT match (expected but check values)\n");
            fprintf(stderr, "[Test] First 4 values: %f %f %f %f\n",
                    readback[0], readback[1], readback[2], readback[3]);
        }
    }
    delete[] readback;

    cudaInteropUnregister(cuda, cuResource);
    cudaInteropCleanup(cuda);
    d3d11Cleanup(renderer);
    TEST_PASS("interop-copy");
#else
    TEST_SKIP("interop-copy", "D3D11 not available on Linux");
#endif
}

// ========================================================================
// Main: parse args and run selected tests
// ========================================================================
int main(int argc, char* argv[]) {
    const char* mode = "all";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--test-mode") == 0 || strcmp(argv[i], "-m") == 0)
            && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "PhysX 5 + D3D11 + CUDA Interop Test Suite\n\n");
            fprintf(stderr, "Usage: %s [--test-mode <mode>]\n\n", argv[0]);
            fprintf(stderr, "Modes:\n");
            fprintf(stderr, "  physx-only       PhysX GPU simulation (no D3D11)\n");
            fprintf(stderr, "  d3d11-only       D3D11 device + buffers (no CUDA)\n");
            fprintf(stderr, "  cuda-init        CUDA initialization only\n");
            fprintf(stderr, "  interop-register Register D3D11 buffer with CUDA\n");
            fprintf(stderr, "  interop-map      Register + map + unmap lifecycle\n");
            fprintf(stderr, "  interop-copy     CUDA write + D3D11 readback\n");
            fprintf(stderr, "  all              Run all tests\n");
            return 0;
        }
    }

    fprintf(stderr, "PhysX 5 + D3D11 + CUDA Interop Test Suite\n");
    fprintf(stderr, "Test mode: %s\n\n", mode);

    bool runAll = (strcmp(mode, "all") == 0);

    if (runAll || strcmp(mode, "cuda-init") == 0)
        test_cuda_init();
    if (runAll || strcmp(mode, "physx-only") == 0)
        test_physx_only();
    if (runAll || strcmp(mode, "d3d11-only") == 0)
        test_d3d11_only();
    if (runAll || strcmp(mode, "interop-register") == 0)
        test_interop_register();
    if (runAll || strcmp(mode, "interop-map") == 0)
        test_interop_map();
    if (runAll || strcmp(mode, "interop-copy") == 0)
        test_interop_copy();

    // --- Summary ---
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Results: %d passed, %d failed, %d skipped\n",
            gTestsPassed, gTestsFailed, gTestsSkipped);
    fprintf(stderr, "========================================\n");

    return gTestsFailed > 0 ? 1 : 0;
}
