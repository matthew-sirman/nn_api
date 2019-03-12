#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

class NN_LIB_API timer
{
public:
	timer();
	~timer();

	void start();
	void stop();
	void reset() { total_time = 0; }

	float elapsed();

	void stamp();
private:
	cudaEvent_t g_t1, g_t2;

	float total_time = 0;
};

