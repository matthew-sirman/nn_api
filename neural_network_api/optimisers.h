#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#include "instruction_functions.h"
#include "optimiser_kernel.h"

namespace nn {
	class NN_LIB_API optimiser
	{
	public:
		optimiser();
		~optimiser();
		virtual void optimise(trainable_function * t_func, int t_step) = 0;
	};

	class NN_LIB_API stochastic_gradient_descent : public optimiser
	{
	public:
		stochastic_gradient_descent(float learning_rate);
		stochastic_gradient_descent(float learning_rate, float momentum);
		~stochastic_gradient_descent();

		void optimise(trainable_function * t_func, int t_step) override;

	private:
		float momentum;
		float learning_rate;
	};

	class NN_LIB_API adam : public optimiser
	{
	public:
		adam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
		~adam();

		void optimise(trainable_function * t_func, int t_step) override;

	private:
		double learning_rate;
		double beta1;
		double beta2;
		double epsilon;
	};
}
