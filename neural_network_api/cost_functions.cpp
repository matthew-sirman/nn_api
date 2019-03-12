#include "stdafx.h"

#include "cost_functions.h"

namespace nn {
	
	cost_function::cost_function()
		: cost_function(true)
	{

	}

	cost_function::cost_function(bool one_hot)
	{
		this->input_shape = input_shape;
		this->one_hot = one_hot;
	}

	cost_function::~cost_function()
	{

	}

	void cost_function::initialise(shape input_shape, size_t batch_size, int total_size)
	{
		this->batch_size = batch_size;
		this->input_shape = input_shape;
		this->total_size = total_size;
		//allocate_device_pointer(&d_output, batch_size);
		allocate_device_pointer(&d_output, 1);
		allocate_device_pointer(&d_der_vector, batch_size * input_shape.size());
		allocate_device_pointer(&d_avg_loss, 1);
	}

	void cost_function::uninitialise()
	{
		deallocate_device_pointer(d_output);
		deallocate_device_pointer(d_der_vector);
		deallocate_device_pointer(d_avg_loss);
	}

	float cost_function::get_average_loss()
	{
		return avg_loss;
	}

	float * cost_function::get_derivative_vector()
	{
		return d_der_vector;
	}

	size_t cost_function::get_size()
	{
		return input_shape.size();
	}

	void squared_error::cost(float * input, float * y, size_t batch_size)
	{
		squared_error_cost(input, y, d_output, input_shape.size() * batch_size);

		average_value(d_output, d_avg_loss, batch_size);
		cudaMemcpy(&avg_loss, d_avg_loss, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	}

	void squared_error::cost_derivative(float * input, float * y, size_t batch_size)
	{
		squared_error_cost_derivative(input, y, d_der_vector, input_shape.size() * batch_size);
	}

	void softmax_cross_entropy::cost(float * input, float * y, size_t batch_size)
	{
		cuda_safe_call(cudaMemset(d_output, 0, sizeof(float)));
		softmax_cross_entropy_cost(input, y, d_output, input_shape.size(), batch_size);

		float * loss = new float();
		cuda_safe_call(cudaMemcpy(loss, d_output, sizeof(float), cudaMemcpyDeviceToHost));

		avg_loss = *loss / batch_size;

		/*average_value(d_output, d_avg_loss, batch_size, total_size);

		float tmp;
		cudaMemcpy(&tmp, d_avg_loss, sizeof(float) * 1, cudaMemcpyDeviceToHost);

		avg_loss += tmp;/**/
	}

	void softmax_cross_entropy::cost_derivative(float * input, float * y, size_t batch_size)
	{
		apply_softmax(input, d_softmax, input_shape.size(), batch_size, 1);

		softmax_cross_entropy_derivative(d_softmax, y, d_der_vector, input_shape.size(), batch_size);
	}

	void softmax_cross_entropy::initialise(shape input_shape, size_t batch_size, int n_batches)
	{
		cost_function::initialise(input_shape, batch_size, n_batches);
		allocate_device_pointer(&d_softmax, input_shape.size() * batch_size);
	}

	void softmax_cross_entropy::uninitialise()
	{
		cost_function::uninitialise();
		deallocate_device_pointer(d_softmax);
	}
}
