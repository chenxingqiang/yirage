#ifndef CUDA_RUNTIME_COMPAT_H
#define CUDA_RUNTIME_COMPAT_H

/*
 * CUDA runtime compatibility header for non-CUDA builds
 * Provides basic CUDA type definitions when CUDA is not available
 */

#ifdef __CUDACC__
// If CUDA is available, include the real headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
// CUDA compatibility stubs for non-CUDA builds

#ifdef __cplusplus
extern "C" {
#endif

// Basic CUDA types
typedef int cudaError_t;

#ifdef __cplusplus
struct dim3 {
    unsigned int x, y, z;
    
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) 
        : x(vx), y(vy), z(vz) {}
};

// uint2, uint3, uint4 are defined in vector_types.h
#else
typedef struct { unsigned int x, y, z; } dim3;
typedef struct { unsigned int x, y; } uint2;
typedef struct { unsigned int x, y, z; } uint3;
typedef struct { unsigned int x, y, z, w; } uint4;
#endif

// CUDA error constants
#define cudaSuccess 0
#define cudaErrorInvalidValue 1

// CUDA function stubs
static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }
static inline cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
static inline cudaError_t cudaSetDevice(int device) { (void)device; return cudaSuccess; }
static inline cudaError_t cudaGetDevice(int *device) { *device = 0; return cudaSuccess; }

// Memory management stubs
static inline cudaError_t cudaMalloc(void **devPtr, size_t size) {
    *devPtr = malloc(size);
    return *devPtr ? cudaSuccess : cudaErrorInvalidValue;
}

static inline cudaError_t cudaFree(void *devPtr) {
    free(devPtr);
    return cudaSuccess;
}

static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
    (void)kind; // Ignore copy direction
    memcpy(dst, src, count);
    return cudaSuccess;
}

// Stream and event stubs
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

static inline cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    *stream = NULL;
    return cudaSuccess;
}

static inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

static inline cudaError_t cudaEventCreate(cudaEvent_t *event) {
    *event = NULL;
    return cudaSuccess;
}

static inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    (void)event;
    return cudaSuccess;
}

// Helper functions
#ifdef __cplusplus
static inline dim3 make_dim3(unsigned int x, unsigned int y = 1, unsigned int z = 1) {
    return dim3(x, y, z);
}

static inline uint2 make_uint2(unsigned int x, unsigned int y = 0) {
    uint2 result;
    result.x = x;
    result.y = y;
    return result;
}

static inline uint3 make_uint3(unsigned int x, unsigned int y = 0, unsigned int z = 0) {
    uint3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

static inline uint4 make_uint4(unsigned int x, unsigned int y = 0, unsigned int z = 0, unsigned int w = 0) {
    uint4 result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    return result;
}
#endif

#ifdef __cplusplus
}
#endif

// CUDA kernel launch compatibility
#ifndef __CUDACC__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__

// Kernel launch syntax replacement
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shmem, stream, ...) \
    do { \
        (void)grid; (void)block; (void)shmem; (void)stream; \
        /* kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__) becomes a no-op */ \
    } while(0)
#endif

#endif /* __CUDACC__ */

#endif /* CUDA_RUNTIME_COMPAT_H */
