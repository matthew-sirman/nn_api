#include "stdafx.h"

#include "timer.h"

void timer::start()
{
	//create Cuda events for the start and stop times
	cudaEventCreate(&g_t1);
	cudaEventCreate(&g_t2);

	//record that the timer has started
	cudaEventRecord(g_t1, 0);
}

void timer::stop()
{
	//record that the timer has stopped
	cudaEventRecord(g_t2, 0);

	//synchronise the event
	cudaEventSynchronize(g_t2);

	//get the elapsed time between the two events
	cudaEventElapsedTime(&total_time, g_t1, g_t2);

	//destroy the events as they are finished with
	cudaEventDestroy(g_t1);
	cudaEventDestroy(g_t2);
}

float timer::elapsed()
{
	return total_time;
}

void timer::stamp()
{
	//default message stamp to console
	printf("Time to complete: %lf ms \n", total_time);
}
