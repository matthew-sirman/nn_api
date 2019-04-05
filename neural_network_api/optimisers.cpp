#include "stdafx.h"

#include "optimisers.h"


namespace nn {
	optimiser::optimiser()
	{
	}

	optimiser::~optimiser()
	{
	}

	stochastic_gradient_descent::stochastic_gradient_descent(float learning_rate)
		: stochastic_gradient_descent(learning_rate, 0)
	{
	}

	stochastic_gradient_descent::stochastic_gradient_descent(float learning_rate, float momentum)
	{
		this->learning_rate = learning_rate;
		this->momentum = momentum;
	}

	stochastic_gradient_descent::~stochastic_gradient_descent()
	{
	}

	void stochastic_gradient_descent::optimise(trainable_function * t_func, int t_step)
	{
		throw new exception("Not implemented");
		//t_func->train_function(learning_rate, momentum);
	}

	adam::adam(double learning_rate, double beta1, double beta2, double epsilon)
	{
		this->learning_rate = learning_rate;
		this->beta1 = beta1;
		this->beta2 = beta2;
		this->epsilon = epsilon;
	}

	adam::~adam()
	{
	}

	void adam::optimise(trainable_function * t_func, int t_step)
	{
		float * mom_p = t_func->get_momentum();
		float * vel_p = t_func->get_velocity();

		size_t size = t_func->get_train_tensor_size();

		adam_update_momentum(
			mom_p,
			t_func->get_derivative_vector(),
			beta1,
			size
		);

		adam_update_velocity(
			vel_p,
			t_func->get_derivative_vector(),
			beta2,
			size
		);

		adam_update_parameters(
			t_func->get_train_vector(),
			mom_p,
			vel_p,
			learning_rate,
			beta1,
			beta2,
			epsilon,
			t_step,
			size
		);

		/*float * test = (float *)malloc(sizeof(float) * 50);
		cudaMemcpy(test, &t_func->get_train_vector()[5 * 5 * 32], sizeof(float) * 50, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 50; i++)
			printf("Test[%d] = %e\n", i, test[i]);
		printf("\n");/**/
	}
}