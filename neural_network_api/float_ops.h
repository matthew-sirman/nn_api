#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <cuda.h>
#include <cuda_runtime.h>

//float2 functions
extern NN_LIB_API inline __host__ __device__ float2 operator+(float2 a, float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

extern NN_LIB_API inline __host__ __device__ void operator+=(float2 &a, float2 b) {
	a.x += b.x; a.y += b.y;
}

extern NN_LIB_API inline __host__ __device__ float2 operator*(float2 a, float2 b) {
	return make_float2(a.x * b.x, a.y * b.y);
}

extern NN_LIB_API inline __host__ __device__ void operator*=(float2 &a, float2 b) {
	a.x *= b.x; a.y *= b.y;
}

//float4 functions
extern NN_LIB_API inline __host__ __device__ float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

extern NN_LIB_API inline __host__ __device__ void operator+=(float4 &a, float4 b) {
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

extern NN_LIB_API inline __host__ __device__ float4 operator*(float4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

extern NN_LIB_API inline __host__ __device__ void operator*=(float4 &a, float4 b) {
	a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}