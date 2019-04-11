#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace nnet {
	//Timer
	//This class can be used to time events. Call start() to begin timing
	//and stop() to end timing. Call elapsed() to get the time elapsed between
	//the two calls 
	class timer
	{
	public:
		//Default Constructor
		timer() { g_t1 = NULL; g_t2 = NULL; }

		//Destructor
		~timer() {};

		//Start
		//Call this function to start timing
		void start();

		//Start
		//Call this function to stop timing
		void stop();

		//Elapsed
		//Returns the elapsed time between starting and stopping
		//the timer
		float elapsed();

		//Stamp
		//Print a default timestamp message to the console
		void stamp();
	private:
		//cuda events representing the start and stop times
		cudaEvent_t g_t1, g_t2;

		//the total time between starting and stopping
		float total_time = 0;
	};

}