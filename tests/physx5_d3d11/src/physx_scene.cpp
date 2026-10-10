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

// PhysX 5 GPU scene setup — PBD particle simulation
// Based on SnippetPBDCloth from the official PhysX 5 SDK
//
// This creates a simple particle system that runs on the GPU via CUDA.
// PhysX internally calls cuInit, cuCtxCreate, cuModuleLoadData, cuLaunchKernel, etc.
// When ZLUDA is active, these calls go through ZLUDA → HIP → AMD GPU.

#include "physx_scene.h"
#include <cstdio>
#include <cstring>

// PhysX headers
#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"

using namespace physx;

// Error callback
class TestErrorCallback : public PxErrorCallback {
public:
    void reportError(PxErrorCode::Enum code, const char* message,
                     const char* file, int line) override {
        const char* errorStr = "UNKNOWN";
        switch (code) {
            case PxErrorCode::eDEBUG_INFO:    errorStr = "INFO"; break;
            case PxErrorCode::eDEBUG_WARNING: errorStr = "WARN"; break;
            case PxErrorCode::eINVALID_PARAMETER: errorStr = "INVALID_PARAM"; break;
            case PxErrorCode::eINVALID_OPERATION: errorStr = "INVALID_OP"; break;
            case PxErrorCode::eOUT_OF_MEMORY: errorStr = "OOM"; break;
            case PxErrorCode::eINTERNAL_ERROR: errorStr = "INTERNAL"; break;
            case PxErrorCode::eABORT:         errorStr = "ABORT"; break;
            case PxErrorCode::ePERF_WARNING:  errorStr = "PERF"; break;
            default: break;
        }
        fprintf(stderr, "[PhysX %s] %s (%s:%d)\n", errorStr, message, file, line);
    }
};

static TestErrorCallback gErrorCallback;
static PxDefaultAllocator gAllocator;

bool physxInit(PhysXScene& s, uint32_t numParticles) {
    fprintf(stderr, "[PhysX] Initializing with %u particles...\n", numParticles);

    // --- Foundation ---
    s.foundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    if (!s.foundation) {
        fprintf(stderr, "[PhysX] ERROR: PxCreateFoundation failed\n");
        return false;
    }

    // --- Physics ---
    s.physics = PxCreatePhysics(PX_PHYSICS_VERSION, *s.foundation,
                                 PxTolerancesScale(), true, nullptr);
    if (!s.physics) {
        fprintf(stderr, "[PhysX] ERROR: PxCreatePhysics failed\n");
        return false;
    }
    PxInitExtensions(*s.physics, nullptr);

    // --- CUDA Context Manager ---
    // This is where PhysX calls cuInit() and cuCtxCreate() internally.
    // With ZLUDA: cuInit → hipInit, cuCtxCreate → hipCtxCreate
    PxCudaContextManagerDesc cudaDesc;
    s.cudaContextManager = PxCreateCudaContextManager(*s.foundation, cudaDesc,
                                                       PxGetProfilerCallback());
    if (!s.cudaContextManager || !s.cudaContextManager->contextIsValid()) {
        fprintf(stderr, "[PhysX] ERROR: CUDA context creation failed!\n");
        fprintf(stderr, "[PhysX] This means cuInit/cuCtxCreate failed through ZLUDA.\n");
        return false;
    }
    fprintf(stderr, "[PhysX] CUDA context created successfully\n");

    // --- Scene with GPU enabled ---
    PxSceneDesc sceneDesc(s.physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cudaContextManager = s.cudaContextManager;
    sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    sceneDesc.solverType = PxSolverType::eTGS;
    sceneDesc.cpuDispatcher = PxDefaultCpuDispatcherCreate(2);
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;

    s.scene = s.physics->createScene(sceneDesc);
    if (!s.scene) {
        fprintf(stderr, "[PhysX] ERROR: Scene creation failed\n");
        return false;
    }
    fprintf(stderr, "[PhysX] GPU-enabled scene created\n");

    // --- PBD Particle System ---
    // This loads PhysX GPU CUDA kernels (PTX compiled at runtime)
    s.maxParticles = numParticles;
    s.numParticles = numParticles;

    PxPBDParticleSystem* pbd = s.physics->createPBDParticleSystem(*s.cudaContextManager,
                                                                    numParticles);
    if (!pbd) {
        fprintf(stderr, "[PhysX] ERROR: PBD particle system creation failed\n");
        fprintf(stderr, "[PhysX] This likely means GPU kernel compilation failed.\n");
        return false;
    }

    // Basic PBD parameters
    pbd->setRestOffset(0.05f);
    pbd->setContactOffset(0.05f + 0.01f);
    pbd->setParticleContactOffset(0.05f);
    pbd->setSolidRestOffset(0.05f);

    s.scene->addActor(*pbd);
    s.particleSystem = pbd;

    // --- Create particle buffer with initial positions ---
    // Particles in a grid for cloth-like behavior
    PxParticleAndDiffuseBuffer* particleBuffer = nullptr;
    {
        PxParticleBufferDesc bufferDesc;
        bufferDesc.maxParticles = numParticles;
        bufferDesc.numActiveParticles = numParticles;

        // Create buffer (this allocates device memory)
        PxCudaContext* cudaContext = s.cudaContextManager->getCudaContext();
        // PhysX manages device memory internally
        particleBuffer = s.physics->createParticleAndDiffuseBuffer(
            numParticles, 0, *s.cudaContextManager);
    }

    if (particleBuffer) {
        pbd->addParticleBuffer(particleBuffer);
        fprintf(stderr, "[PhysX] Added %u particles to PBD system\n", numParticles);
    }

    fprintf(stderr, "[PhysX] Initialization complete\n");
    return true;
}

void physxStep(PhysXScene& s, float dt) {
    if (!s.scene) return;
    s.scene->simulate(dt);
    s.scene->fetchResults(true);
    // For PBD particle systems, also fetch particle results
    s.scene->fetchResultsParticleSystem();
}

void* physxGetParticlePositionsDevice(PhysXScene& s) {
    // PhysX stores particle positions in GPU memory.
    // Access via the direct GPU API if available.
    return s.deviceParticlePositions;
}

void physxCleanup(PhysXScene& s) {
    fprintf(stderr, "[PhysX] Cleaning up...\n");
    if (s.scene) {
        s.scene->release();
        s.scene = nullptr;
    }
    if (s.physics) {
        PxCloseExtensions();
        s.physics->release();
        s.physics = nullptr;
    }
    if (s.cudaContextManager) {
        s.cudaContextManager->release();
        s.cudaContextManager = nullptr;
    }
    if (s.foundation) {
        s.foundation->release();
        s.foundation = nullptr;
    }
    fprintf(stderr, "[PhysX] Cleanup complete\n");
}
