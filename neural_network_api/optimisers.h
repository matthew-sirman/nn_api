#pragma once

#include "graph.h"
#include "instruction_functions.h"
#include "optimiser_kernel.h"

using namespace nnet::instructions;

namespace nnet {
	namespace optimisers {
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
			virtual void optimise() = 0;

			//API FUNCTION
			//Initialise
			//Initialise the optimiser by passing in the graph it has to optimise
			virtual void initialise(network_graph* g) { this->g = g; t_funcs = g->get_train_functions(); };

			//API FUNCTION
			//Uninitialise
			//Uninitialise the optimiser
			virtual void uninitialise() {};
		protected:
			//the graph this optimiser has to optimise
			network_graph* g;

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
			void optimise() override;

		private:
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
			~adam() {};

			//API FUNCTION
			//Optimiser
			//Called by the API to optimise the variables in the training functions after
			//each training step.
			void optimise() override;

			//API FUNCTION
			//Initialise
			//Initialise the optimiser by passing in the training functions it has to optimise
			void initialise(network_graph* g) override;

			//API FUNCTION
			//Uninitialise
			//Uninitialise the optimiser
			void uninitialise() override;

		private:
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
