#pragma once

//define macros for thread syncing, warp syncing
//and atomic addition
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

//constants for use in kernel

//BLOCK_SIZE
//The default maximum number of threads per block used
//for the majority of simple kernels in the API
constexpr auto BLOCK_SIZE = 128;

//SOFTMAX_MAX_CLASSES
//The maximum number of classes the Softmax function can handle
constexpr auto SOFTMAX_MAX_CLASSES = 2048;

//MAX_BATCH_SIZE
//The largest batch size kernels can handle
constexpr auto MAX_BATCH_SIZE = 2048;

//Cuda Safe Call
//Macro for capturing and handling exceptions raised by Cuda
//functions
#define cuda_safe_call(err) __cuda_safe_call(err, __FILE__, __LINE__)

//API FUNCTION
//Cuda Safe Call
//Handle Cuda exceptions
inline void __cuda_safe_call(cudaError error, const char *file, int line, bool abort = true) {
	//if anything other than success
	if (error != cudaSuccess) {
		//print message with line, file and error
		printf("cuda_safe_call failed in file %s on line %d with error %s\n", file, line, cudaGetErrorString(error));
		//by default, exit the application
		if (abort)
			exit(error);
	}
}
