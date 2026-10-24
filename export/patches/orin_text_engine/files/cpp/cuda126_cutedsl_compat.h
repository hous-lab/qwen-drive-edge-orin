// Compatibility shim for CUDA 12.6 lacking CuTe-DSL's CUDA 12.8+ runtime module API.
// Force-included via -include so every translation unit sees the typedef + decl.
// cudaLibraryUnload is provided at link time by cutedsl_cuda_runtime_library_shim.c.
#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#if !defined(cudaLibrary_t)
typedef CUlibrary cudaLibrary_t;
#endif
extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
