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
		//initialise all the default parameters for the cost function
		this->batch_size = batch_size;
		this->input_shape = input_shape;

		//allocate pointers for device memory
		allocate_device_float_pointer(&d_output, 1);
		allocate_device_float_pointer(&d_der_vector, batch_size * input_shape.size());
		allocate_device_float_pointer(&d_avg_loss, 1);
	}

	void cost_function::uninitialise()
	{
		//deallocate pointers for device memory
		deallocate_device_float_pointer(d_output);
		deallocate_device_float_pointer(d_der_vector);
		deallocate_device_float_pointer(d_avg_loss);
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
		//run the cost function and write the cost to the output
		squared_error_cost(input, y, d_output, input_shape.size() * batch_size);

		//find the average value from the array returned
		average_value(d_output, d_avg_loss, batch_size);

		//copy the loss into host memory
		cuda_safe_call(cudaMemcpy(&avg_loss, d_avg_loss, sizeof(float) * 1, cudaMemcpyDeviceToHost));
	}

	void squared_error::cost_derivative(float * input, float * y, size_t batch_size)
	{
		//get the derivative between the distributions
		squared_error_cost_derivative(input, y, d_der_vector, input_shape.size() * batch_size);
	}

	void softmax_cross_entropy::cost(float * input, float * y, size_t batch_size)
	{
		//initalise the output to 0
		cuda_safe_call(cudaMemset(d_output, 0, sizeof(float)));

		//calculate the cross entropy cost for the distributions
		softmax_cross_entropy_cost(input, y, d_output, input_shape.size(), batch_size);

		//get the total loss
		float * loss = new float();
		cuda_safe_call(cudaMemcpy(loss, d_output, sizeof(float), cudaMemcpyDeviceToHost));

		//divide total loss by size for average loss
		avg_loss = *loss / batch_size;
	}

	void softmax_cross_entropy::cost_derivative(float * input, float * y, size_t batch_size)
	{
		//initialise the softmax and derivative vectors to 0
		cuda_safe_call(cudaMemset(d_softmax, 0, sizeof(float) * input_shape.size() * batch_size));
		cuda_safe_call(cudaMemset(d_der_vector, 0, sizeof(float) * input_shape.size() * batch_size));

		//get the softmax output of the input vector
		apply_softmax(input, d_softmax, input_shape.size(), batch_size, 1);

		//calucate the derivative
		softmax_cross_entropy_derivative(d_softmax, y, d_der_vector, input_shape.size(), batch_size);

		//average the derivative at this point to prevent explosions and save on computations
		scalar_matrix_multiply_f(d_der_vector, d_der_vector, 1.0 / batch_size, input_shape.size() * batch_size);
	}

	void softmax_cross_entropy::initialise(shape input_shape, size_t batch_size, int n_batches)
	{
		//call the base class initialiser
		cost_function::initialise(input_shape, batch_size, n_batches);
		//allocate the softmax pointer
		allocate_device_float_pointer(&d_softmax, input_shape.size() * batch_size);
	}

	void softmax_cross_entropy::uninitialise()
	{
		//call the base class uninitialiser
		cost_function::uninitialise();
		//deallocate the softmax pointer
		deallocate_device_float_pointer(d_softmax);
	}
}
