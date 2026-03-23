#pragma once

// Minimal D3D11 renderer for PhysX particle visualization
// Creates a D3D11 device and vertex buffer that can be shared with CUDA.
// Under DXVK, D3D11 calls are translated to Vulkan.

#include <cstdint>

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

struct D3D11Renderer {
#ifdef _WIN32
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;

    // Shared vertex buffer for particle positions (CUDA writes, D3D11 reads)
    ID3D11Buffer* particleVB = nullptr;

    // Staging buffer for CPU readback (validation)
    ID3D11Buffer* stagingBuffer = nullptr;
#endif

    uint32_t maxParticles = 0;
    bool initialized = false;
    bool headless = false;  // true = no window, just device + buffers
};

// Initialize D3D11 device.
// If headless=true, creates device without a window (for validation only).
// If headless=false, creates a window + swap chain for visual rendering.
bool d3d11Init(D3D11Renderer& renderer, uint32_t maxParticles, bool headless);

// Create a vertex buffer that can be shared with CUDA via interop.
// Returns the D3D11 buffer pointer (cast to void* for cross-platform).
void* d3d11CreateSharedBuffer(D3D11Renderer& renderer, uint32_t numFloat4);

// Read back buffer contents to CPU memory for validation.
// Returns true if readback succeeded, data written to outData.
bool d3d11ReadbackBuffer(D3D11Renderer& renderer, void* outData, uint32_t sizeBytes);

// Render one frame (particle points). Only works if not headless.
void d3d11RenderFrame(D3D11Renderer& renderer, uint32_t numParticles);

// Clean up all D3D11 resources.
void d3d11Cleanup(D3D11Renderer& renderer);
