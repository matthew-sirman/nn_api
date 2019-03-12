#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include "instructions_kernel.h"
#include "tensor.h"
#include "instruction_functions.h"

namespace nn {
	class NN_LIB_API cost_function
	{
	public:
		cost_function();
		cost_function(bool one_hot);
		~cost_function();

		virtual void cost(float * input, float * y, size_t batch_size) = 0;
		virtual void cost_derivative(float * input, float * y, size_t batch_size) = 0;

		virtual void initialise(shape input_shape, size_t batch_size, int total_size);
		virtual void uninitialise();

		void set_total_size(size_t total_size) { this->total_size = total_size; }

		virtual float get_average_loss();

		void clear_loss() { avg_loss = 0; }

		float * get_derivative_vector();

		size_t get_size();

	protected:
		shape input_shape;
		//size_t targets_size;
		size_t batch_size;
		size_t total_size;

		float * d_output;
		float * d_der_vector;

		bool one_hot;

		float * d_avg_loss;
		float avg_loss = 0;
	};

	class NN_LIB_API squared_error : public cost_function
	{
	public:
		squared_error() : cost_function::cost_function() {};
		squared_error(bool one_hot) : cost_function::cost_function(one_hot) {};
		~squared_error() {};

		void cost(float * input, float * y, size_t batch_size) override;
		void cost_derivative(float * input, float * y, size_t batch_size) override;
	};

	class NN_LIB_API softmax_cross_entropy : public cost_function
	{
	public:
		softmax_cross_entropy() : cost_function::cost_function() {};
		~softmax_cross_entropy() {};

		void cost(float * input, float * y, size_t batch_size) override;
		void cost_derivative(float * input, float * y, size_t batch_size) override;

		void initialise(shape input_shape, size_t batch_size, int total_size) override;
		void uninitialise() override;
	private:
		float * d_softmax;
	};
}


