#pragma once

namespace nnet {
	//Variable Initialiser
	//Use the variable initialiser to specify a mean and standard deviation
	//for random normal sampling. Avoids creating random tensors manually
	//by allowing the helper functions to know the distribution to sample from
	struct variable_initialiser
	{
	public:
		//Constructor specifying mean and standard deviation for initialising
		//a normal distribution
		variable_initialiser(float mean = 0.0f, float stddev = 0.01f) {
			this->mean = mean;
			this->stddev = stddev;
		}

		//Destructor
		~variable_initialiser() {};

		//Mean
		//The mean of the normal distribution the variable will be sampled
		//from
		float mean;

		//Standard Deviation
		//The standard deviation of the normal distribution the variable will be 
		//sampled from
		float stddev;
	};
}

