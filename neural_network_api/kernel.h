#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#ifdef __CUDACC__
#define cuda_syncthreads() __syncthreads()
#define cuda_syncwarp() __syncwarp()
#define atomic_add(arr, val) atomicAdd(arr, val)
#else
#define cuda_syncthreads()
#define cuda_syncwarp()
#define atomic_add(arr, val)
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "util.h"

constexpr auto BLOCK_SIZE = 128;
constexpr auto SOFTMAX_MAX_CLASSES = 2048;
constexpr auto MAX_BATCH_SIZE = 2048;

#define cuda_safe_call(err) __cuda_safe_call(err, __FILE__, __LINE__)

extern NN_LIB_API inline void __cuda_safe_call(cudaError error, const char *file, int line, bool abort = true) {
	if (error != cudaSuccess) {
		printf("cuda_safe_call failed in file %s on line %d with error %s\n", file, line, cudaGetErrorString(error));
		if (abort)
			exit(error);
	}
}
