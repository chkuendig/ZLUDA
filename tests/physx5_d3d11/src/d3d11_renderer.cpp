// Minimal D3D11 renderer — creates device, shared buffers, and optional rendering.
// Under Wine+DXVK, D3D11CreateDevice → Vulkan device creation.

#include "d3d11_renderer.h"
#include <cstdio>
#include <cstring>

#ifdef _WIN32

bool d3d11Init(D3D11Renderer& r, uint32_t maxParticles, bool headless) {
    fprintf(stderr, "[D3D11] Initializing (headless=%d, maxParticles=%u)...\n",
            headless, maxParticles);

    r.maxParticles = maxParticles;
    r.headless = headless;

    // --- Create D3D11 Device ---
    // Under DXVK, this creates a Vulkan device internally.
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL featureLevel;
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr;
    if (headless) {
        // No swap chain — just device + context
        hr = D3D11CreateDevice(
            nullptr,                    // default adapter
            D3D_DRIVER_TYPE_HARDWARE,   // hardware device (DXVK intercepts this)
            nullptr,                    // no software module
            flags,
            featureLevels, 3,
            D3D11_SDK_VERSION,
            &r.device,
            &featureLevel,
            &r.context
        );
    } else {
        // TODO: Create window + swap chain for visual rendering
        // For now, just create headless
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            featureLevels, 3, D3D11_SDK_VERSION,
            &r.device, &featureLevel, &r.context
        );
    }

    if (FAILED(hr)) {
        fprintf(stderr, "[D3D11] ERROR: D3D11CreateDevice failed (hr=0x%08lx)\n", hr);
        fprintf(stderr, "[D3D11] Under Wine: is DXVK installed in the prefix?\n");
        return false;
    }

    fprintf(stderr, "[D3D11] Device created (feature level: 0x%x)\n", featureLevel);
    r.initialized = true;
    return true;
}

void* d3d11CreateSharedBuffer(D3D11Renderer& r, uint32_t numFloat4) {
    if (!r.device) return nullptr;

    uint32_t sizeBytes = numFloat4 * 4 * sizeof(float); // float4 = 16 bytes

    // Create vertex buffer with D3D11_RESOURCE_MISC_SHARED flag.
    // This flag tells DXVK to create a Vulkan buffer that can be exported
    // via VK_KHR_external_memory, which CUDA/HIP can then import.
    D3D11_BUFFER_DESC bufDesc = {};
    bufDesc.ByteWidth = sizeBytes;
    bufDesc.Usage = D3D11_USAGE_DEFAULT;
    bufDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
    bufDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    HRESULT hr = r.device->CreateBuffer(&bufDesc, nullptr, &r.particleVB);
    if (FAILED(hr)) {
        fprintf(stderr, "[D3D11] ERROR: CreateBuffer (shared VB) failed (hr=0x%08lx)\n", hr);
        // Try without MISC_SHARED — may still work for testing lifecycle
        bufDesc.MiscFlags = 0;
        hr = r.device->CreateBuffer(&bufDesc, nullptr, &r.particleVB);
        if (FAILED(hr)) {
            fprintf(stderr, "[D3D11] ERROR: CreateBuffer (fallback) also failed\n");
            return nullptr;
        }
        fprintf(stderr, "[D3D11] WARNING: Created buffer WITHOUT shared flag\n");
    }

    // Create staging buffer for CPU readback
    D3D11_BUFFER_DESC stagingDesc = {};
    stagingDesc.ByteWidth = sizeBytes;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    hr = r.device->CreateBuffer(&stagingDesc, nullptr, &r.stagingBuffer);
    if (FAILED(hr)) {
        fprintf(stderr, "[D3D11] WARNING: Staging buffer creation failed (readback disabled)\n");
    }

    fprintf(stderr, "[D3D11] Created shared buffer (%u bytes, %u float4s)\n",
            sizeBytes, numFloat4);
    return static_cast<void*>(r.particleVB);
}

bool d3d11ReadbackBuffer(D3D11Renderer& r, void* outData, uint32_t sizeBytes) {
    if (!r.context || !r.particleVB || !r.stagingBuffer) return false;

    // Copy GPU buffer → staging buffer
    r.context->CopyResource(r.stagingBuffer, r.particleVB);

    // Map staging buffer for CPU read
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    HRESULT hr = r.context->Map(r.stagingBuffer, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        fprintf(stderr, "[D3D11] ERROR: Map staging buffer failed (hr=0x%08lx)\n", hr);
        return false;
    }

    memcpy(outData, mapped.pData, sizeBytes);
    r.context->Unmap(r.stagingBuffer, 0);
    return true;
}

void d3d11RenderFrame(D3D11Renderer& r, uint32_t numParticles) {
    if (!r.context || r.headless) return;
    // TODO: Implement visual rendering (draw particles as point sprites)
    // For now, this is a placeholder — the important part is the interop path.
    (void)numParticles;
}

void d3d11Cleanup(D3D11Renderer& r) {
    fprintf(stderr, "[D3D11] Cleaning up...\n");
    if (r.stagingBuffer) { r.stagingBuffer->Release(); r.stagingBuffer = nullptr; }
    if (r.particleVB)    { r.particleVB->Release();    r.particleVB = nullptr; }
    if (r.swapChain)     { r.swapChain->Release();     r.swapChain = nullptr; }
    if (r.context)       { r.context->Release();       r.context = nullptr; }
    if (r.device)        { r.device->Release();        r.device = nullptr; }
    r.initialized = false;
    fprintf(stderr, "[D3D11] Cleanup complete\n");
}

#else // !_WIN32

// Linux stubs — D3D11 tests only run under Wine
bool d3d11Init(D3D11Renderer&, uint32_t, bool) {
    fprintf(stderr, "[D3D11] Not available on Linux — run under Wine/Proton\n");
    return false;
}
void* d3d11CreateSharedBuffer(D3D11Renderer&, uint32_t) { return nullptr; }
bool d3d11ReadbackBuffer(D3D11Renderer&, void*, uint32_t) { return false; }
void d3d11RenderFrame(D3D11Renderer&, uint32_t) {}
void d3d11Cleanup(D3D11Renderer&) {}

#endif
