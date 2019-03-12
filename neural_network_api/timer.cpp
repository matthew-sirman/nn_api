#include "stdafx.h"

#include "timer.h"

timer::timer()
{
}

timer::~timer()
{
}

void timer::start()
{
	cudaEventCreate(&g_t1);
	cudaEventCreate(&g_t2);
	cudaEventRecord(g_t1, 0);
}

void timer::stop()
{
	cudaEventRecord(g_t2, 0);
	cudaEventSynchronize(g_t2);

	cudaEventElapsedTime(&total_time, g_t1, g_t2);

	cudaEventDestroy(g_t1);
	cudaEventDestroy(g_t2);
}

float timer::elapsed()
{
	return total_time;
}

void timer::stamp()
{
	printf("Time to complete: %lf ms \n", total_time);
}
