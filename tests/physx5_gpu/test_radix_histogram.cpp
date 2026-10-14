// Open-source NVIDIA PhysX-derived test source.
// BSD 3-Clause License
// 
// Copyright (c) 2008-2025, NVIDIA Corporation
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//    
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//    
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Standalone HIP reproducer for the PhysX radix sort histogram bug.
// Loads the ZLUDA-compiled ELF module and runs radixSortMultiBlockLaunchWithCount
// with the same parameters that PhysX uses (27 elements, shift=0).
//
// Build: ext/rocm/bin/hipcc tests/physx5_gpu/test_radix_histogram.cpp -o /tmp/test_radix -I ext/rocm/include
// Run:   LD_LIBRARY_PATH=ext/rocm/lib /tmp/test_radix

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { hipError_t err = (x); if (err != hipSuccess) { fprintf(stderr, "HIP error %d at %s:%d: %s\n", err, __FILE__, __LINE__, hipGetErrorString(err)); exit(1); } } while(0)

// PxgRadixSortDesc (from PxgRadixSortDesc.h)
struct PxgRadixSortDesc {
    uint32_t* inputKeys;       // offset 0
    uint32_t* inputRanks;      // offset 8
    uint32_t* outputKeys;      // offset 16
    uint32_t* outputRanks;     // offset 24
    uint32_t* radixBlockCounts; // offset 32
    uint32_t  count;           // offset 40
    uint32_t  _pad;            // offset 44 (padding to 48 bytes)
};

int main() {
    hipDevice_t dev;
    hipCtx_t ctx;
    CHECK(hipInit(0));
    CHECK(hipDeviceGet(&dev, 0));
    CHECK(hipCtxCreate(&ctx, 0, dev));

    // Load the ELF module
    hipModule_t module;
    const char* elf_path = "/tmp/zluda_ir_dump/module_7_elf";
    FILE* f = fopen(elf_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", elf_path); return 1; }
    fseek(f, 0, SEEK_END);
    size_t elf_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* elf_data = malloc(elf_size);
    fread(elf_data, 1, elf_size, f);
    fclose(f);
    CHECK(hipModuleLoadData(&module, elf_data));
    free(elf_data);

    hipFunction_t kernel;
    CHECK(hipModuleGetFunction(&kernel, module, "radixSortMultiBlockLaunchWithCount"));
    printf("Loaded kernel: radixSortMultiBlockLaunchWithCount\n");

    // Allocate buffers
    const uint32_t count = 27;
    const uint32_t padded = 28; // ceil(27/4)*4
    const uint32_t num_blocks = 32;
    const uint32_t shift = 0;

    uint32_t* d_inputKeys;
    uint32_t* d_inputRanks;
    uint32_t* d_outputKeys;
    uint32_t* d_outputRanks;
    uint32_t* d_radixBlockCounts;
    PxgRadixSortDesc* d_desc;

    CHECK(hipMalloc(&d_inputKeys, padded * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_inputRanks, padded * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_outputKeys, padded * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_outputRanks, padded * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_radixBlockCounts, 16 * num_blocks * sizeof(uint32_t)));
    CHECK(hipMalloc(&d_desc, sizeof(PxgRadixSortDesc)));

    // Initialize input keys: [0xFFFFFFFF, 0*18, 1, 2, 3, 4, 5, 6, 7, 8, 0xFFFFFFFF(sentinel)]
    uint32_t h_keys[28];
    h_keys[0] = 0xFFFFFFFF;
    for (int i = 1; i <= 18; i++) h_keys[i] = 0;
    for (int i = 19; i <= 26; i++) h_keys[i] = i - 18; // 1-8
    h_keys[27] = 0xFFFFFFFF; // sentinel

    // Initialize ranks: [0, 1, 2, ..., 26, 0xFFFFFFFF(sentinel)]
    uint32_t h_ranks[28];
    for (int i = 0; i < 27; i++) h_ranks[i] = i;
    h_ranks[27] = 0xFFFFFFFF;

    CHECK(hipMemcpyHtoD(d_inputKeys, h_keys, padded * sizeof(uint32_t)));
    CHECK(hipMemcpyHtoD(d_inputRanks, h_ranks, padded * sizeof(uint32_t)));
    CHECK(hipMemset(d_outputKeys, 0, padded * sizeof(uint32_t)));
    CHECK(hipMemset(d_outputRanks, 0, padded * sizeof(uint32_t)));
    CHECK(hipMemset(d_radixBlockCounts, 0, 16 * num_blocks * sizeof(uint32_t)));

    // Set up descriptor
    PxgRadixSortDesc h_desc;
    h_desc.inputKeys = d_inputKeys;
    h_desc.inputRanks = d_inputRanks;
    h_desc.outputKeys = d_outputKeys;
    h_desc.outputRanks = d_outputRanks;
    h_desc.radixBlockCounts = d_radixBlockCounts;
    h_desc.count = count;
    h_desc._pad = 0;
    CHECK(hipMemcpyHtoD(d_desc, &h_desc, sizeof(h_desc)));

    // Launch kernel
    void* args[] = { &d_desc, (void*)&count, (void*)&shift };
    printf("Launching with grid=[%d,1,1] block=[1024,1,1] count=%d shift=%d\n", num_blocks, count, shift);
    CHECK(hipModuleLaunchKernel(kernel, num_blocks, 1, 1, 1024, 1, 1, 0, 0, args, nullptr));
    CHECK(hipDeviceSynchronize());
    printf("Kernel completed\n");

    // Read back histogram
    uint32_t h_rbc[16 * 32];
    CHECK(hipMemcpyDtoH(h_rbc, d_radixBlockCounts, 16 * num_blocks * sizeof(uint32_t)));

    // Print digit-major layout: [digit * numBlocks + blockIdx]
    printf("Digit-major (block 0):\n");
    uint32_t total = 0;
    for (int d = 0; d < 16; d++) {
        uint32_t val = h_rbc[d * num_blocks + 0];
        if (val != 0) printf("  digit %2d: %u\n", d, val);
        total += val;
    }
    printf("  total: %u (expected 28)\n", total);

    // Print block-major layout: [blockIdx * 16 + digit]
    printf("Block-major (block 0):\n");
    total = 0;
    for (int d = 0; d < 16; d++) {
        uint32_t val = h_rbc[0 * 16 + d];
        if (val != 0) printf("  digit %2d: %u\n", d, val);
        total += val;
    }
    printf("  total: %u\n", total);

    // Print ALL nonzero entries
    printf("All nonzero entries:\n");
    for (int i = 0; i < 16 * (int)num_blocks; i++) {
        if (h_rbc[i] != 0) printf("  [%d] = %u (digit=%d block=%d)\n", i, h_rbc[i], i/num_blocks, i%num_blocks);
    }

    // Cleanup
    hipFree(d_inputKeys);
    hipFree(d_inputRanks);
    hipFree(d_outputKeys);
    hipFree(d_outputRanks);
    hipFree(d_radixBlockCounts);
    hipFree(d_desc);
    hipModuleUnload(module);
    hipCtxDestroy(ctx);
    return 0;
}
