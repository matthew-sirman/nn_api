#pragma once

#include "instructions_kernel.h"
#include "tensor.h"
#include "instruction_functions.h"

namespace nnet {
	namespace cost_functions {
		//Abstract base class for all cost functions
		//Gives a metric for how far two distributions are from each other
		class cost_function : public operation
		{
		public:
			//Default constructor
			cost_function();

			//Constructor with one_hot flag option
			cost_function(bool one_hot);

			//Destructor
			~cost_function();

			//Abstract method for returning the derivative between the distributions x (observed) 
			//and y (expected)
			virtual void cost_derivative() = 0;

			//Initialise the cost function
			virtual void initialise(shape input_shape, size_t batch_size);

			//Uninitialise the cost function
			virtual void uninitialise();

			//Return the average loss for the last batch
			virtual float get_average_loss();

			//Reset the average loss value to 0
			void clear_loss() { avg_loss = 0; }

			//Return the derivate vector after calculating
			float* get_derivative_vector();

			//Returns the input size of the cost function
			size_t get_size();

			//Get Input Placeholder
			//Return a reference to the input placeholder. This can be used to
			//feed data into this operation
			placeholder& get_input_placeholder() { return input_data_ph; }

			//Get Taget Placeholder
			//Return a reference to the target placeholder. This can be used to
			//feed data into this operation
			placeholder& get_target_placeholder() { return target_data_ph; }

		protected:
			//shape of the input
			shape input_shape;

			//the batch size
			size_t batch_size = 0;

			//a float variable to hold the cost output
			float* d_output = nullptr;

			//float array to hold the derivative vector
			float* d_der_vector = nullptr;

			//flag to indicate whether the labels are in one-hot format
			bool one_hot = false;

			//float to store the average loss on the device
			float* d_avg_loss = nullptr;

			//float to store the average loss on the host
			float avg_loss = 0;

			//placeholder for the input data
			placeholder input_data_ph = placeholder("inputs");

			//placeholder for the target data
			placeholder target_data_ph = placeholder("targets");
		};

		//Squared Error Cost Function
		//Calculates the squared error between two distributions
		//C(x, y) = Sum[(x - y)^2]
		//This class should not be directly instantiated. Use
		//the set_cost_function template method of the
		//network_model class
		class squared_error : public cost_function
		{
		public:
			//Default Constructor
			squared_error() : cost_function::cost_function() {};
			//Constructor with one-hot flag specifier
			squared_error(bool one_hot) : cost_function::cost_function(one_hot) {};

			//Destructor
			~squared_error() {};

			//API FUNCTION
			//Run
			//Returns the squared error cost or loss between two distributions
			void run() override;

			//API FUNCTION
			//Cost Derivative
			//Returns the derivative between the "input" and "y" distributions
			void cost_derivative() override;
		};

		//Cross Entropy Cost Function
		//Calculates the cross between two distributions
		//C(x, y) = -Sum[ylog(x)]
		//This class should not be directly instantiated. Use
		//the set_cost_function template method of the
		//network_model class
		class softmax_cross_entropy : public cost_function
		{
		public:
			//Default Constructor
			softmax_cross_entropy() : cost_function::cost_function() {};

			//Destructor
			~softmax_cross_entropy() {};

			//API FUNCTION
			//Run
			//Returns the cross entropy cost or loss between two distributions
			void run() override;

			//API FUNCTION
			//Cost Derivative
			//Returns the derivative between the "input" and "y" distributions
			void cost_derivative() override;

			//API FUNCTION
			//Initialise
			void initialise(shape input_shape, size_t batch_size) override;

			//API FUNCTION
			//Uninitialise
			void uninitialise() override;
		private:
			//placeholder variable for the softmax function
			float* d_softmax = nullptr;
		};
	}
}


