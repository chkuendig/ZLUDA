#pragma once

// PhysX 5 GPU scene setup
// Based on SnippetPBDCloth from the official PhysX 5 SDK

#include <cstdint>

// Forward declarations — avoid pulling in PhysX headers in every file
namespace physx {
    class PxFoundation;
    class PxPhysics;
    class PxScene;
    class PxCudaContextManager;
    class PxPBDParticleSystem;
}

struct PhysXScene {
    physx::PxFoundation* foundation = nullptr;
    physx::PxPhysics* physics = nullptr;
    physx::PxScene* scene = nullptr;
    physx::PxCudaContextManager* cudaContextManager = nullptr;
    physx::PxPBDParticleSystem* particleSystem = nullptr;

    // Particle data on device (CUDA pointers)
    void* deviceParticlePositions = nullptr;
    uint32_t numParticles = 0;
    uint32_t maxParticles = 0;
};

// Initialize PhysX with GPU support. Returns false if GPU init fails.
bool physxInit(PhysXScene& scene, uint32_t numParticles);

// Step the simulation forward by dt seconds.
void physxStep(PhysXScene& scene, float dt);

// Get the CUDA device pointer to particle positions (float4 array).
// Returns nullptr if not available.
void* physxGetParticlePositionsDevice(PhysXScene& scene);

// Clean up all PhysX resources.
void physxCleanup(PhysXScene& scene);
