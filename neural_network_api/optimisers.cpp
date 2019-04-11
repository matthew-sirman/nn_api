#include "stdafx.h"

#include "optimisers.h"


namespace nnet {
	namespace optimisers {
		stochastic_gradient_descent::stochastic_gradient_descent(float learning_rate, float momentum)
		{
			//setup variables from inputs
			this->learning_rate = learning_rate;
			this->momentum = momentum;
		}

		void stochastic_gradient_descent::optimise()
		{
			throw new exception("Not implemented");
		}

		adam::adam(double learning_rate, double beta1, double beta2, double epsilon)
		{
			//setup variables from inputs
			this->learning_rate = learning_rate;
			this->beta1 = beta1;
			this->beta2 = beta2;
			this->epsilon = epsilon;
		}

		void adam::optimise()
		{
			//loop through each trainable function
			for (int f = 0; f < t_funcs.size(); f++) {
				//get a refernce to the current function
				trainable_function* t_func = t_funcs[f];

				//if the function is locked, continue to the next function
				if (t_func->locked())
					continue;

				//get a reference to the momentum for this function
				float* mom_p = momentum_ps[f];

				//get a reference to the velocity for this function
				float* vel_p = velocity_ps[f];

				//get the size of the training tensor for this function
				size_t size = t_func->get_train_tensor_size();

				//update the momentum
				adam_update_momentum(
					mom_p,
					t_func->get_train_derivative_vector(),
					beta1,
					size
				);

				//update the velocity
				adam_update_velocity(
					vel_p,
					t_func->get_train_derivative_vector(),
					beta2,
					size
				);

				//update the trainable parameters using the momentum and velocity
				//calculated for this training step
				adam_update_parameters(
					t_func->get_train_vector(),
					mom_p,
					vel_p,
					learning_rate,
					beta1,
					beta2,
					epsilon,
					__step,
					size
				);
			}

			//increment the step counter
			__step++;
		}
		void adam::initialise(vector<trainable_function*> t_funcs)
		{
			//call the base initialiser
			optimiser::initialise(t_funcs);

			//loop through each function
			for (int f = 0; f < t_funcs.size(); f++) {
				//declare pointers for momentum and velocity
				float* m_p;
				float* v_p;

				//allocate device memory for momentum and velocity
				allocate_device_float_pointer(&m_p, t_funcs[f]->get_train_tensor_size());
				allocate_device_float_pointer(&v_p, t_funcs[f]->get_train_tensor_size());

				//add the new pointers to a vector to store them both
				momentum_ps.push_back(m_p);
				velocity_ps.push_back(v_p);
			}
		}
		void adam::uninitialise()
		{
			//uninitialise the base function
			optimiser::uninitialise();

			//loop through each function and dereference the momentum and 
			//velocity pointers for each
			for (int f = 0; f < t_funcs.size(); f++) {
				deallocate_device_float_pointer(momentum_ps[f]);
				deallocate_device_float_pointer(velocity_ps[f]);
			}
		}
	}
}