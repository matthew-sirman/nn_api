#include "stdafx.h"

#include "instruction_functions.h"

namespace nn {

	size_t serialisable_function::get_serialise_size()
	{
		return sizeof(shape) * 2;
	}

	void serialisable_function::serialise(char * stream_buff, size_t offset)
	{
		/*
		FORMAT FOR SERIALISED FUNCTION
		Function type

		Base:
		Input size
		Output size

		IF:

		TF:
		Train tensor dims
		Train tensor shape
		Train tensor
		*/
		memcpy(&stream_buff[offset], input_shape.serialise(), sizeof(shape));
		memcpy(&stream_buff[offset + sizeof(shape)], output_shape.serialise(), sizeof(shape));
	}

	void serialisable_function::deserialise(char * stream_buff, size_t offset)
	{
		input_shape.deserialise(stream_buff, offset);
		output_shape.deserialise(stream_buff, offset + sizeof(shape));
	}

	instruction_function::~instruction_function()
	{
		if (initialised)
			uninitialise();
	}

	void instruction_function::initialise()
	{
		instruction_function::initialise(1);
	}

	void instruction_function::initialise(size_t batch_size)
	{
		allocate_device_pointer(&d_out_vector, output_shape.size() * batch_size);
		this->batch_size = batch_size;

		initialised = true;
	}

	void instruction_function::uninitialise()
	{
		deallocate_device_pointer(d_out_vector);

		initialised = false;
	}

	size_t instruction_function::get_serialise_size()
	{
		return serialisable_function::get_serialise_size() + sizeof(function_id);
	}

	void instruction_function::__serialise(char * stream_buffer, size_t offset, function_id func_id)
	{
		memcpy(&stream_buffer[offset], &func_id, sizeof(function_id));
		serialisable_function::serialise(stream_buffer, offset + sizeof(function_id));
	}

	void instruction_function::deserialise(char * stream_buffer, size_t offset)
	{
		serialisable_function::deserialise(stream_buffer, offset + sizeof(function_id));
	}

	float * instruction_function::get_out_vector()
	{
		return d_out_vector;
	}

	float * instruction_function::get_derivative_vector()
	{
		return d_der_vector;
	}
		
	trainable_function::trainable_function(tensor t) 
		: train_tensor(t), derivatives(t.get_shape())
	{
		derivatives = tensor::zeros(t.get_shape());
		type |= instruction_function_type::TRAINABLE;
	}

	void trainable_function::initialise()
	{
		initialise(1);
	}

	void trainable_function::initialise(size_t batch_size)
	{
		instruction_function::initialise(batch_size);
		train_tensor.initialise();
		derivatives.initialise();
	}

	void trainable_function::uninitialise()
	{
		instruction_function::uninitialise();
		deallocate_device_pointer(d_pder_vector);
		train_tensor.uninitialise();
		derivatives.uninitialise();
	}

	size_t trainable_function::get_serialise_size()
	{
		return instruction_function::get_serialise_size() + sizeof(size_t) + train_tensor.get_dimensions() * sizeof(size_t) + sizeof(float) * get_train_tensor_size();
	}

	void trainable_function::__serialise(char * stream_buffer, size_t offset, function_id func_id)
	{
		instruction_function::__serialise(stream_buffer, offset, func_id);
		size_t i_f_offset = instruction_function::get_serialise_size() + offset;

		train_tensor.serialise(stream_buffer, i_f_offset);
	}

	void trainable_function::deserialise(char * stream_buffer, size_t offset)
	{
		instruction_function::deserialise(stream_buffer, offset);
		size_t i_f_offset = instruction_function::get_serialise_size() + offset;

		train_tensor.deserialise(stream_buffer, i_f_offset);
		derivatives = tensor::zeros(train_tensor.get_shape());
	}

	void output_function::initialise()
	{
		initialise({ 1, 1 }, 1);
	}

	void output_function::initialise(shape input_shape, size_t batch_size)
	{
		allocate_device_pointer(&d_out_vector, input_shape.size() * batch_size);
		this->batch_size = batch_size;
		this->input_shape = input_shape;

		initialised = true;
	}

	void output_function::uninitialise()
	{
		deallocate_device_pointer(d_out_vector);

		initialised = false;
	}

	float * output_function::get_out_vector()
	{
		return d_out_vector;
	}

	add_function::add_function(size_t bias_size)
		: trainable_function(tensor(bias_size))
	{
		this->input_shape = { bias_size, 1 };
		this->output_shape = { bias_size, 1 };
	}

	add_function::add_function(tensor biases)
		: trainable_function(biases)
	{
		this->input_shape = { biases.get_shape()[0], 1 };
		this->output_shape = { biases.get_shape()[0], 1 };
	}

	add_function::~add_function()
	{
		instruction_function::~instruction_function();
	}

	void add_function::run(float* input)
	{
		run(input, 1);
	}

	void add_function::run(float * input, int batch_size)
	{
		add_matrices(input, d_out_vector, train_tensor.get_dev_pointer(), input_shape.width, batch_size);
	}

	void add_function::run_derivative(float* input)
	{

	}

	void add_function::run_train_derivative(float * input, int batch_size)
	{

	}

	void add_function::back_propagate(float * current_pds, int num)
	{

	}

	void add_function::initialise()
	{
		initialise(1);
	}

	void add_function::initialise(size_t batch_size)
	{
		trainable_function::initialise(batch_size);

		allocate_device_pointer(&d_der_vector, batch_size * input_shape.width);
		allocate_device_pointer(&d_pder_vector, batch_size * input_shape.width);
		allocate_device_pointer(&d_derivatives, batch_size * input_shape.width);
		allocate_device_pointer(&d_momentum, input_shape.width);
		allocate_device_pointer(&d_velocity, input_shape.width);
		allocate_device_pointer(&d_avg_derivatives, input_shape.width);

		fill_device_array(d_momentum, 0.0, input_shape.width);
		fill_device_array(d_velocity, 0.0, input_shape.width);

		fill_device_array(d_der_vector, 1, batch_size * input_shape.width);
		fill_device_array(d_pder_vector, 1, batch_size * input_shape.width);
	}

	void add_function::uninitialise() {
		trainable_function::uninitialise();
		deallocate_device_pointer(d_der_vector);
		deallocate_device_pointer(d_derivatives);
		deallocate_device_pointer(d_momentum);
		deallocate_device_pointer(d_velocity);
		deallocate_device_pointer(d_avg_derivatives);
	}

	void add_function::avg_partial_derivatives(float * current_pds, int num)
	{
		hadamard_product(current_pds, d_pder_vector, d_derivatives, input_shape.width * num);

		average_vector(d_derivatives, derivatives.get_dev_pointer(), input_shape.width, num, num);

		//add_matrices(d_avg_derivatives, derivatives.get_dev_pointer(), derivatives.get_dev_pointer(), input_shape.width, 1);

		/*float * test = (float *)malloc(sizeof(float) * 10);
		cudaMemcpy(test, derivatives.get_dev_pointer(), sizeof(float) * 10, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 10; i++)
			printf("test[%d] = %e\n", i, test[i]);
		printf("\n");*/
	}
	
	void add_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::ADD);
	}

	/*void add_function::train_function(float learning_rate, float momentum)
	{
		subtract_partial_derivatives(train_tensor.get_dev_pointer(), derivatives.get_dev_pointer(), input_shape.width, learning_rate);
	}*/

	mul_function::mul_function(size_t weight_rows, size_t weight_cols)
		: trainable_function(tensor({ weight_rows, weight_cols }))
	{
		this->input_shape = { weight_cols, 1 };
		this->output_shape = { weight_rows, 1 };
	}

	mul_function::mul_function(tensor weights)
		: trainable_function(tensor(weights))
	{
		this->input_shape = { weights.get_shape()[1], 1 };
		this->output_shape = { weights.get_shape()[0], 1 };
	}

	mul_function::~mul_function()
	{
		instruction_function::~instruction_function();
	}

	void mul_function::run(float* input)
	{
		run(input, 1);
	}

	void mul_function::run(float * input, int batch_size)
	{
		//multiply_matrices(input, d_out_vector, d_partial_mul_outputs, train_tensor.get_dev_pointer(), output_size, input_size, batch_size);
		//multiply_matrices(train_tensor.get_dev_pointer(), input, d_partial_mul_outputs, d_out_vector, output_size, input_size, input_size, batch_size);

		//matrix_multiply(train_tensor.get_dev_pointer(), input, d_out_vector, batch_size, output_size, input_size);

		d_matrix<float> B(batch_size, (int)input_shape.width, input);

		matrix_multiply<float, order::ROW, order::COL, order::COL>(
			d_mat, 
			B, 
			d_out_vec
		);

		/*float * test = (float *)malloc(sizeof(float) * 4);
		cudaMemcpy(test, d_out_vec.d_data, sizeof(float) * 4, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 4; i++)
			printf("mul[%d] = %f\n", i, test[i]);*/
	}

	void mul_function::run_derivative(float* input)
	{

	}

	void mul_function::run_train_derivative(float * input, int batch_size)
	{
		copy_into_device_array(input, d_pder_vector, input_shape.width * batch_size, 0);
		//transpose(d_pder_vector, d_pder_vector, batch_size, input_size);
	}

	void mul_function::back_propagate(float * current_pds, int num)
	{

		//Try creating a constructor
		d_matrix<float> A({ (int)output_shape.width, (int)input_shape.width, get_train_vector() });
		d_matrix<float> B({ (int)num, (int)output_shape.width, current_pds });
		d_matrix<float> C({ (int)num, (int)input_shape.width, d_bp_temp });

		matrix_multiply<float, order::COL, order::COL, order::COL>(
			A,
			B,
			C
		);

		//cuda_safe_call(cudaMemcpy(d_current_layer_cr_derivative, d_temp, sizeof(float) * current_batch_size * i_func->input_size, cudaMemcpyDeviceToDevice));

		/*float * test = (float *)malloc(sizeof(float) * input_shape.width * num);
		cudaMemcpy(test, d_bp_temp, sizeof(float) * input_shape.width * num, cudaMemcpyDeviceToHost);

		for (int m = 0; m < 4; m++) {
			for (int n = 0; n < 4; n++) {
				printf("%.32g ", test[m * num + n]);
			}
			printf("\n");
		}
		printf("\n\n");*/

		copy_into_device_array(d_bp_temp, current_pds, num * input_shape.width, 0);

		/*float * test = (float *)malloc(sizeof(float) * 10);
		cudaMemcpy(test, current_pds, sizeof(float) * 10, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 10; i++)
			printf("test[%d] = %e\n", i, test[i]);
		printf("\n");*/
	}

	void mul_function::initialise()
	{
		initialise(1);
	}

	void mul_function::initialise(size_t batch_size)
	{
		trainable_function::initialise(batch_size);
		//d_transpose_matrix = train_tensor.get_transpose().get_dev_pointer();
		//allocate_device_pointer(&d_partial_mul_outputs, output_size * batch_size * ceil_div(M_TILE_SIZE_X, input_size));
		allocate_device_pointer(&d_pder_vector, input_shape.width * batch_size);
		allocate_device_pointer(&d_derivatives, input_shape.width * output_shape.width);
		allocate_device_pointer(&d_momentum, input_shape.width * output_shape.width);
		allocate_device_pointer(&d_velocity, input_shape.width * output_shape.width);
		allocate_device_pointer(&d_avg_derivatives, input_shape.width * output_shape.width);
		allocate_device_pointer(&d_bp_temp, input_shape.width * batch_size);
		//allocate_device_pointer(&current_pds_t, input_size * batch_size);

		d_der_vector = train_tensor.get_dev_pointer();//train_tensor.get_transpose().get_dev_pointer();

		fill_device_array(d_momentum, 0.0, input_shape.width * output_shape.width);
		fill_device_array(d_velocity, 0.0, input_shape.width * output_shape.width);

		create_device_matrix(d_mat, input_shape.width, output_shape.width, train_tensor.get_dev_pointer());
		create_device_matrix(d_out_vec, batch_size, output_shape.width, d_out_vector);
	}

	void mul_function::uninitialise()
	{
		trainable_function::uninitialise();
		//deallocate_device_pointer(d_partial_mul_outputs);
		deallocate_device_pointer(d_derivatives);
		deallocate_device_pointer(d_momentum);
		deallocate_device_pointer(d_velocity);
		deallocate_device_pointer(d_avg_derivatives);
		deallocate_device_pointer(d_bp_temp);
		//deallocate_device_pointer(current_pds_t);
		//cuda_safe_call(cudaFree(d_mat));
		//cuda_safe_call(cudaFree(d_out_vec));
	}

	void mul_function::avg_partial_derivatives(float * current_pds, int num)
	{
		//allocate another device memory pointer, batch_size * input_size.
		//allocate partial outputs pointer (or maybe just reuse the other? Depends if currently required at this point)

		//d_input_p => new pointer of batch_size * input_size (current_pds)
		//d_out => d_derivatives
		//d_partial_outputs => partial outputs pointer
		//d_weights_p => partial derivatives

		//transpose(current_pds, current_pds_t, num, output_size);

		//multiply_matrices(current_pds_t, d_pder_vector, d_partial_mul_outputs, d_derivatives, output_size, num, num, input_size);

		//matrix_multiply(current_pds, d_pder_vector, d_derivatives, input_size, output_size, num);

		//transpose(d_derivatives, d_derivatives, input_size, output_size);

		d_matrix<float> A({ num, (int)output_shape.width, current_pds });
		d_matrix<float> B({ (int)input_shape.width, num, d_pder_vector });
		d_matrix<float> C({ (int)input_shape.width, (int)output_shape.width, d_derivatives });

		matrix_multiply<float, order::COL, order::ROW, order::ROW>(
			A,
			B,
			C
		);

		scalar_matrix_multiply_f(d_derivatives, derivatives.get_dev_pointer(), 1.0f / num, input_shape.width * output_shape.width);
		//add_matrices(derivatives.get_dev_pointer(), derivatives.get_dev_pointer(), d_derivatives, input_shape.width * output_shape.width, 1);

		/*float * test = (float *)malloc(sizeof(float) * 10);
		cudaMemcpy(test, d_derivatives, sizeof(float) * 10, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 10; i++)
			printf("test[%d] = %e\n", i, test[i]);
		printf("\n");*/

		/*float * test = (float *)malloc(sizeof(float) * 4);
		cudaMemcpy(test, derivatives.get_dev_pointer(), sizeof(float) * 4, cudaMemcpyDeviceToHost);

		for (int i = 0; i < 4; i++)
			printf("Test[%d] = %f\n", i, test[i]);*/
	}
	
	void mul_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::MUL);
	}

	/*void mul_function::train_function(float learning_rate, float momentum)
	{
		subtract_partial_derivatives(train_tensor.get_dev_pointer(), derivatives.get_dev_pointer(), input_shape.width * output_shape.width, learning_rate);

		//d_der_vector = train_tensor.get_transpose().get_dev_pointer();
	}*/

	relu_function::relu_function()
		: relu_function(0)
	{

	}

	relu_function::relu_function(size_t input_size)
		: instruction_function()
	{
		this->input_shape = { input_size, 1 };
		this->output_shape = { input_size, 1 };
	}

	relu_function::~relu_function()
	{
		instruction_function::~instruction_function();
	}

	void relu_function::run(float* input)
	{
		run(input, 1);
	}

	void relu_function::run(float * input, int batch_size)
	{
		apply_relu(input, d_out_vector, input_shape.size() * batch_size, 0);
	}

	void relu_function::run_derivative(float* input)
	{
		relu_derivative(input, d_der_vector, input_shape.size() * batch_size, 0);
	}

	void relu_function::back_propagate(float * current_pds, int num)
	{
		hadamard_product(current_pds, get_derivative_vector(), current_pds, input_shape.size() * num);
	}

	void relu_function::initialise()
	{
		initialise(1);
	}

	void relu_function::initialise(size_t batch_size)
	{
		instruction_function::initialise(batch_size);
		allocate_device_pointer(&d_der_vector, input_shape.size() * batch_size);
	}

	void relu_function::uninitialise()
	{
		instruction_function::uninitialise();
	}

	void relu_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::RELU);
	}

	leaky_relu_function::leaky_relu_function(float alpha)
		: leaky_relu_function(0, alpha)
	{

	}

	leaky_relu_function::leaky_relu_function(size_t input_size, float alpha)
		: instruction_function()
	{
		this->input_shape = { input_size, 1 };
		this->output_shape = { input_size, 1 };
		this->alpha = alpha;
	}

	leaky_relu_function::~leaky_relu_function()
	{
		instruction_function::~instruction_function();
	}

	void leaky_relu_function::run(float* input)
	{
		run(input, 1);
	}

	void leaky_relu_function::run(float * input, int batch_size)
	{
		apply_relu(input, d_out_vector, input_shape.size() * batch_size, alpha);
	}

	void leaky_relu_function::run_derivative(float* input)
	{
		relu_derivative(input, d_der_vector, input_shape.size() * batch_size, alpha);
	}

	void leaky_relu_function::back_propagate(float * current_pds, int num)
	{
		hadamard_product(current_pds, get_derivative_vector(), current_pds, input_shape.size() * num);
	}

	void leaky_relu_function::initialise()
	{
		initialise(1);
	}

	void leaky_relu_function::initialise(size_t batch_size)
	{
		instruction_function::initialise(batch_size);
		allocate_device_pointer(&d_der_vector, input_shape.size() * batch_size);
	}

	void leaky_relu_function::uninitialise()
	{
		instruction_function::uninitialise();
	}
	
	size_t leaky_relu_function::get_serialise_size()
	{
		return instruction_function::get_serialise_size() + sizeof(float);
	}

	void leaky_relu_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::L_RELU);
		size_t new_offset = instruction_function::get_serialise_size();
		memcpy(&stream_buffer[new_offset], reinterpret_cast<char*>(reinterpret_cast<void*>(&alpha)), sizeof(float));
	}

	void leaky_relu_function::deserialise(char * stream_buffer, size_t offset)
	{
		instruction_function::deserialise(stream_buffer, offset);
		size_t new_offset = instruction_function::get_serialise_size();
		memcpy(&alpha, reinterpret_cast<float*>(reinterpret_cast<void*>(&stream_buffer[new_offset])), sizeof(float));
	}

	softmax::softmax()
		: softmax(0)
	{

	}

	softmax::softmax(size_t input_size)
	{
		this->input_shape = { input_size, 1 };
		//this->beta = beta;
	}

	/*softmax_function::softmax_function(size_t input_size, float beta)
	{
		this->input_size = input_size;
		this->output_size = input_size;
		this->beta = beta;
	}*/

	softmax::~softmax()
	{
		output_function::~output_function();
	}

	void softmax::run(float * input)
	{
		run(input, 1);
	}

	void softmax::run(float * input, int batch_size)
	{
		apply_softmax(input, d_out_vector, input_shape.width, batch_size, 1);
	}
	
	/*void softmax_function::run_derivative(float * input)
	{
		softmax_derivative(input, d_der_vector, input_size * batch_size, beta);
	}*/

	void softmax::initialise()
	{
		initialise({ 1, 1 }, 1);
	}

	void softmax::initialise(shape input_shape, size_t batch_size)
	{
		output_function::initialise(input_shape, batch_size);
		//allocate_device_pointer(&d_der_vector, input_size * output_size * batch_size);
		//allocate_device_pointer(&d_partial_mul_outputs, )
	}

	void softmax::uninitialise()
	{
		output_function::uninitialise();
	}
	
	batch_normalisation_function::batch_normalisation_function()
		: batch_normalisation_function(0)
	{

	}

	batch_normalisation_function::batch_normalisation_function(size_t input_size)
	{
		this->input_shape = { input_size, 1 };
	}

	void batch_normalisation_function::run(float * input)
	{
		run(input, 1);
	}

	void batch_normalisation_function::run(float * input, int batch_size)
	{
	}

	void batch_normalisation_function::run_derivative(float * input)
	{
	}

	void batch_normalisation_function::back_propagate(float * current_pds, int num)
	{
	}

	void batch_normalisation_function::initialise()
	{
		initialise(1);
	}

	void batch_normalisation_function::initialise(size_t batch_size)
	{
		instruction_function::initialise(batch_size);
	}

	void batch_normalisation_function::uninitialise()
	{
		instruction_function::uninitialise();
	}

	void batch_normalisation_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::BATCH_NORM);
	}

	conv2d_function::conv2d_function(shape input_shape, shape filter_shape, size_t n_filters, shape padding)
		: trainable_function(tensor({ filter_shape.width, filter_shape.height, filter_shape.depth, n_filters }))
	{
		this->filter_shape = filter_shape;
		this->output_shape.depth = n_filters;
		this->padding = padding;
		set_input_shape(input_shape);
	}

	conv2d_function::conv2d_function(tensor filter, shape padding)
		: trainable_function(filter)
	{
		if (filter.get_dimensions() == 4) {
			this->filter_shape = shape(filter.get_shape()[0], filter.get_shape()[1], filter.get_shape()[2]);
			this->output_shape.depth = filter.get_shape()[3];
			this->padding = padding;
		}
		else {
			throw new exception("Conv2d filter must be four dimensional (width, height, depth, filters)");
		}
	}

	conv2d_function::~conv2d_function()
	{
	}

	void conv2d_function::run(float * input)
	{
		run(input, 1);
	}

	void conv2d_function::run(float * input, int batch_size)
	{
		filter_convolve_2d(
			input,
			get_filter().get_dev_pointer(),
			d_out_vector,
			input_shape,
			output_shape,
			filter_shape,
			padding,
			batch_size
		);
	}

	void conv2d_function::run_derivative(float * input)
	{

	}

	void conv2d_function::run_train_derivative(float * input, int num)
	{
		cuda_safe_call(cudaMemcpy(d_pder_vector, input, sizeof(float) * input_shape.size() * num, cudaMemcpyDeviceToDevice));
	}

	void conv2d_function::back_propagate(float * current_pds, int num)
	{
		filter_outer_convolve_2d(current_pds, get_filter().get_dev_pointer(), d_tmp_backprop_output, output_shape, input_shape, filter_shape, padding, num);
		cuda_safe_call(cudaMemcpy(current_pds, d_tmp_backprop_output, sizeof(float) * input_shape.size() * num, cudaMemcpyDeviceToDevice));
	}

	void conv2d_function::initialise()
	{
		initialise(1);
	}

	void conv2d_function::initialise(size_t batch_size)
	{
		trainable_function::initialise(batch_size);
		allocate_device_pointer(&d_tmp_backprop_output, input_shape.size() * batch_size);
		allocate_device_pointer(&d_derivatives, filter_shape.size() * output_shape.depth);
		allocate_device_pointer(&d_pder_vector, input_shape.size() * batch_size);
		allocate_device_pointer(&d_momentum, filter_shape.size() * output_shape.depth);
		allocate_device_pointer(&d_velocity, filter_shape.size() * output_shape.depth);
	}

	void conv2d_function::uninitialise()
	{
		trainable_function::uninitialise();
		deallocate_device_pointer(d_tmp_backprop_output);
		deallocate_device_pointer(d_derivatives);
		deallocate_device_pointer(d_momentum);
		deallocate_device_pointer(d_velocity);
	}

	void conv2d_function::avg_partial_derivatives(float * current_pds, int num)
	{
		//fitler_convolve_2d_derivative(d_pder_vector, current_pds, d_derivatives, input_shape, output_shape, filter_shape, num);
		/*filter_convolve_2d(
			d_pder_vector,
			current_pds,
			d_derivatives,
			output_shape
		);*/

		/*filter_convolve_2d(
			d_pder_vector,
			current_pds,
			d_derivatives,
			input_shape,
			filter_shape,
			output_shape,
			num
		);*/

		filter_convolve_2d_derivative(
			d_pder_vector,
			current_pds,
			d_derivatives,
			input_shape,
			output_shape,
			filter_shape,
			padding,
			num
		);

		//average_vector(d_derivatives, derivatives.get_dev_pointer(), filter_shape.size() * output_shape.depth, num, num);
		scalar_matrix_multiply_f(d_derivatives, derivatives.get_dev_pointer(), 1.0 / num, filter_shape.size() * output_shape.depth);
	}

	size_t conv2d_function::get_serialise_size()
	{
		return trainable_function::get_serialise_size() + sizeof(shape);
	}

	void conv2d_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::CONV_2D);
		size_t new_offset = trainable_function::get_serialise_size() + offset;
		memcpy(&stream_buffer[new_offset], padding.serialise(), sizeof(shape));
	}

	void conv2d_function::deserialise(char * stream_buffer, size_t offset)
	{
		trainable_function::deserialise(stream_buffer, offset);
		vector<size_t> t_shape = train_tensor.get_shape();
		filter_shape = shape(t_shape[0], t_shape[1], t_shape[2]);
		size_t new_offset = trainable_function::get_serialise_size() + offset;
		padding.deserialise(stream_buffer, new_offset);
	}

	void conv2d_function::set_input_shape(shape input_shape)
	{
		if (input_shape.depth != filter_shape.depth) {
			throw new exception("Input depth and filter depth must be equal");
		}
		this->input_shape = input_shape;
		this->output_shape.width = input_shape.width - filter_shape.width + 1 + padding.width * 2;
		this->output_shape.height = input_shape.height - filter_shape.height + 1 + padding.height * 2;
	}

	pool_function::pool_function(shape pool_size, shape stride)
	{
		this->pool_size = pool_size;
		this->stride = stride;
	}

	pool_function::~pool_function()
	{
	}

	void pool_function::run(float * input)
	{
		run(input, 1);
	}

	void pool_function::run(float * input, int batch_size)
	{
		pool_2d(input, d_mask, d_out_vector, input_shape, pool_size, stride, output_shape, batch_size);

		/*float * test = (float *)malloc(sizeof(float) * 144 * 4 * 2);
		cudaMemcpy(test, d_out_vector, sizeof(float) * 144 * 4 * 2, cudaMemcpyDeviceToHost);

		for (int i = 144*4; i < 144*5; i++)
			printf("test[%d] = %f\n", i, test[i]);

		printf("\n");*/
	}

	void pool_function::run_derivative(float * input)
	{

	}

	void pool_function::back_propagate(float * current_pds, int num)
	{
		cudaMemcpy(test, d_mask, sizeof(int) * 10, cudaMemcpyDeviceToHost);
		fill_device_array(d_der_vector, 0, input_shape.size() * num);
		pool_2d_derivative(current_pds, d_mask, d_der_vector, output_shape, input_shape, num);
		cuda_safe_call(cudaMemcpy(current_pds, d_der_vector, sizeof(float) * input_shape.size() * num, cudaMemcpyDeviceToDevice));
		/*if (cudaMemcpy(current_pds, d_der_vector, sizeof(float) * input_shape.size() * num, cudaMemcpyDeviceToDevice) != cudaSuccess) {

			float * test0 = (float *)malloc(sizeof(float) * 10);
			cudaMemcpy(test0, d_out_vector, sizeof(float) * 10, cudaMemcpyDeviceToHost);

			for (int i = 0; i < 10; i++) {
				printf("test[%d] = %d, out[%d] = %d\n", i, test0[i], i, test[i]);
			}
			printf("\n");
		}*/
	}

	void pool_function::initialise()
	{
		initialise(1);
	}

	void pool_function::initialise(size_t batch_size)
	{
		instruction_function::initialise(batch_size);
		cuda_safe_call(cudaMallocManaged(&d_mask, sizeof(int) * input_shape.size() * batch_size));
		allocate_device_pointer(&d_der_vector, input_shape.size() * batch_size);
		test = (int *)malloc(sizeof(int) * 10);
	}

	void pool_function::uninitialise()
	{
		instruction_function::uninitialise();
		cuda_safe_call(cudaFree(d_mask));
		deallocate_device_pointer(d_der_vector);
		free(test);
	}

	void pool_function::set_input_shape(shape input_shape)
	{
		if ((input_shape.width - pool_size.width) % stride.width == 0 &&
			(input_shape.height - pool_size.height) % stride.height == 0 &&
			(input_shape.depth - pool_size.depth) % stride.depth == 0) {
			this->input_shape = input_shape;
			this->output_shape = shape(
				(input_shape.width - pool_size.width) / stride.width + 1,
				(input_shape.height - pool_size.height) / stride.height + 1,
				(input_shape.depth - pool_size.depth) / stride.depth + 1
			);
		}
		else {
			throw new exception("Input shape and pool don't align. Check pool size and stride");
		}
	}

	size_t pool_function::get_serialise_size()
	{
		return instruction_function::get_serialise_size() + sizeof(shape) * 2;
	}

	void pool_function::serialise(char * stream_buffer, size_t offset)
	{
		__serialise(stream_buffer, offset, function_id::POOL);
		size_t new_offset = instruction_function::get_serialise_size() + offset;
		memcpy(&stream_buffer[new_offset], pool_size.serialise(), sizeof(shape));
		memcpy(&stream_buffer[new_offset + sizeof(shape)], stride.serialise(), sizeof(shape));
	}

	void pool_function::deserialise(char * stream_buffer, size_t offset)
	{
		instruction_function::deserialise(stream_buffer, offset);
		size_t new_offset = instruction_function::get_serialise_size() + offset;
		pool_size.deserialise(stream_buffer, new_offset);
		stride.deserialise(stream_buffer, new_offset + sizeof(shape));
	}

}
