#pragma once

#include <functional>

#include "graph.h"
#include "nn_ops.h"
#include "instruction_functions.h"
#include "optimiser_kernel.h"

using namespace std;
using namespace nnet::instructions;

namespace nnet {
	namespace optimisers {
		//Optimiser Operation
		//An operation that will optimise a graph with a certain optimiser when run.
		//Inputs and targets should be fed to the placeholders for this operation
		class optimiser_operation : public operation {
		public:
			//Constructor specifying the graph to optimise, the optimisation function and
			//any metrics to be calculated
			optimiser_operation(network_graph* g, function<void()> opt_fn, int metrics = 0) {
				//assign values from params
				this->g = g;
				this->opt_fn = opt_fn;
				this->metrics = metrics;
			}

			//API FUNCTION
			//Use network_graph.run() instead
			//Run
			//Run this operation by calculating the gradients in the graph
			//and optimising them wit the callback specified
			void run() override {
				//calculate the gradients feeding in the inputs and targets
				calculate_gradients(
					*g, 
					get_placeholder_value(input_data_ph), 
					get_placeholder_value(target_data_ph), 
					batch_size, 
					metrics
				);

				//optimise the network with the callback
				opt_fn();
			}

			//Get Input Placeholder
			//Return a reference to the input placeholder. This can be used to
			//feed data into this operation
			placeholder& get_input_placeholder() { return input_data_ph; }

			//Get Target Placeholder
			//Return a reference to the target placeholder. This can be used to
			//feed data into this operation
			placeholder& get_target_placeholder() { return target_data_ph; }

		private:
			//placeholder for the input data
			placeholder input_data_ph = placeholder("inputs");

			//placeholder for the target data
			placeholder target_data_ph = placeholder("targets");

			//the network graph we are optimising
			network_graph* g = nullptr;

			//the callback actual optimisation function
			function<void()> opt_fn;

			//the metrics to calculate when optimising
			int metrics;
		};

		//Optimiser Base Class
		//Abstract base class for all optimiser functions. To create a custom optimiser,
		//inherit from this class and implement the abstract functions.
		//An optimiser should update a set of trainable parameters given a set of trainable
		//functions.
		class optimiser
		{
		public:
			//Default Constructor
			optimiser() {};

			//Destructor
			~optimiser() {};

			//API FUNCTION
			//Optimiser
			//Called by the API to optimise the variables in the training functions after
			//each training step.
			virtual optimiser_operation* optimise(network_graph* g, int metrics = 0) = 0;
		protected:
			//internal optimise function where we actuall update parameters
			virtual void __optimise() = 0;

			//the trainable functions in the graph
			vector<trainable_function*> t_funcs;
		};

		//NOT IMPLEMENTED
		//Stochastic Gradient Descent Optimiser
		//Optimises a function by taking small steps in the direction of steepest
		//descent to approach a local minimum
		class stochastic_gradient_descent : public optimiser
		{
		public:
			//Constructor specifying the learning rate. Momentum defaults to 0
			stochastic_gradient_descent(float learning_rate) : stochastic_gradient_descent(learning_rate, 0) {};

			//Constructor specifying the learning rate and the momentum
			stochastic_gradient_descent(float learning_rate, float momentum);

			//Destructor
			~stochastic_gradient_descent() {};

			//API FUNCTION
			//Optimiser
			//Called by the API to optimise the variables in the training functions after
			//each training step.
			optimiser_operation* optimise(network_graph* g, int metrics = 0) override;

		private:
			//internal optimise function where we actuall update parameters
			void __optimise() override;

			//constant value of the momentum added to each training stage
			float momentum;

			//constant value for the learning rate to scale the size of steps
			//taken by the optimiser
			float learning_rate;
		};

		//Adam Optimiser
		//Optimises a network model using the Adam algorithm. The Adam algorithm is an 
		//extension on SGD algorithm for training
		class adam : public optimiser
		{
		public:
			//Constructor specifying the learning rate. Decay parameters and epsilon offset have default values
			//so do not need to be explicitly set
			adam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

			//Destructor
			~adam();

			//API FUNCTION
			//Optimiser
			//Called by the API to optimise the variables in the training functions after
			//each training step.
			optimiser_operation* optimise(network_graph* g, int metrics = 0) override;

		private:
			//internal optimise function where we actuall update parameters
			void __optimise() override;

			//constant value for the learning rate to scale the size of steps
			//taken by the optimiser
			double learning_rate;

			//decay rate for the momentum vector
			double beta1;

			//decay rate for the velocity vector
			double beta2;

			//small offset to avoid division by 0
			double epsilon;

			//vector of momentium placeholders for each trainable layer
			vector<float*> momentum_ps;

			//vector of velocity placeholders for each trainable layer
			vector<float*> velocity_ps;

			//step counter for decay
			int __step = 0;
		};
	}
}
