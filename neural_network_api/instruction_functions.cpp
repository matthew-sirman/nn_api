#include "stdafx.h"

#include "instruction_functions.h"

namespace nnet {
	namespace instructions {
		size_t serialisable_function::get_serialise_size()
		{
			//serialisable function alone holds 2 "shape" variables
			return sizeof(shape) * 2;
		}

		void serialisable_function::serialise(char* stream_buff, size_t offset)
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
			//copy input and output into
			memcpy(&stream_buff[offset], input_shape.serialise(), sizeof(shape));
			memcpy(&stream_buff[offset + sizeof(shape)], output_shape.serialise(), sizeof(shape));
		}

		void serialisable_function::deserialise(char* stream_buff, size_t offset)
		{
			//get the input and output shapes from the stream buffer
			input_shape.deserialise(stream_buff, offset);
			output_shape.deserialise(stream_buff, offset + sizeof(shape));
		}

		instruction_function::~instruction_function()
		{
			//uninitialise the function if deleted
			if (initialised)
				uninitialise();
		}

		void instruction_function::initialise(size_t batch_size)
		{
			//allocate the output vector pointer
			allocate_device_float_pointer(&d_out_vector, output_shape.size() * batch_size);

			//allocate the derivative output vector pointer
			allocate_device_float_pointer(&d_out_derivatives, input_shape.size() * batch_size);

			//flag that this function is initialised fully
			initialised = true;
		}

		void instruction_function::uninitialise()
		{
			//destroy the output vector pointer and free the memory
			deallocate_device_float_pointer(d_out_vector);

			//destroy the derivative output vector pointer
			deallocate_device_float_pointer(d_out_derivatives);

			//flag this function is uninitialised
			initialised = false;
		}

		size_t instruction_function::get_serialise_size()
		{
			//returns the base function size and adds on the size of the function id flag which will be written out
			return serialisable_function::get_serialise_size() + sizeof(function_id);
		}

		void instruction_function::__serialise(char* stream_buffer, size_t offset, function_id func_id)
		{
			//serialised first the function id (so deserialisation knows how to begin deserialising)
			memcpy(&stream_buffer[offset], &func_id, sizeof(function_id));

			//serialises the base function after the id flag into the buffer
			serialisable_function::serialise(stream_buffer, offset + sizeof(function_id));
		}

		void instruction_function::deserialise(char* stream_buffer, size_t offset)
		{
			//calls the base function with the buffer (there is no need to deserialise the id as by this
			//point we already know the function type)
			serialisable_function::deserialise(stream_buffer, offset + sizeof(function_id));
		}

		float* instruction_function::get_out_vector()
		{
			return d_out_vector;
		}

		float* instruction_function::get_derivative_vector()
		{
			return d_out_derivatives;
		}

		trainable_function::trainable_function(tensor t)
			: train_tensor(t), derivatives(t.get_shape())
		{
			//initialise the derivatives tensor to be 0s with the same shape as the
			//training tensor
			derivatives = tensor::zeros(t.get_shape());
		}

		void trainable_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//initialise the tensors
			train_tensor.initialise();
			derivatives.initialise();
		}

		void trainable_function::uninitialise()
		{
			//call the base initialiser
			instruction_function::uninitialise();

			//deallocate the partial derivative pointer
			deallocate_device_float_pointer(d_pder_vector);

			//uninitialise the tensors
			train_tensor.uninitialise();
			derivatives.uninitialise();
		}

		size_t trainable_function::get_serialise_size()
		{
			//return the base size plus the size of the tensor and the a size_t for each dimension
			return instruction_function::get_serialise_size() + sizeof(size_t) + train_tensor.get_dimensions() * sizeof(size_t) + sizeof(float) * get_train_tensor_size();
		}

		void trainable_function::__serialise(char* stream_buffer, size_t offset, function_id func_id)
		{
			//serialise the base function
			instruction_function::__serialise(stream_buffer, offset, func_id);
			size_t i_f_offset = instruction_function::get_serialise_size() + offset;

			//serialise the training tensor into the buffer
			train_tensor.serialise(stream_buffer, i_f_offset);
		}

		void trainable_function::deserialise(char* stream_buffer, size_t offset)
		{
			//deserialise the base function
			instruction_function::deserialise(stream_buffer, offset);
			size_t i_f_offset = instruction_function::get_serialise_size() + offset;

			//deserialise the training tensor from the buffer
			train_tensor.deserialise(stream_buffer, i_f_offset);

			//setup the derivatives tensor to 0s
			derivatives = tensor::zeros(train_tensor.get_shape());
		}

		add_function::add_function(shape bias_size)
			: trainable_function(tensor({ bias_size.width, bias_size.height, bias_size.depth }))
		{
			//set the input and output shapes to the bias size (as the add function
			//doesn't change the shape)
			this->input_shape = bias_size;
			this->output_shape = bias_size;
		}

		add_function::add_function(tensor biases)
			: trainable_function(biases)
		{
			//set the input and output shapes to the bias tensor size (as the add function
			//doesn't change the shape)
			this->input_shape = shape(biases.get_size());
			this->output_shape = shape(biases.get_size());
		}

		add_function::~add_function()
		{
			//destroy the base function
			instruction_function::~instruction_function();
		}

		void add_function::run()
		{
			//add the input matrix to the train tensor matrix
			add_matrices(feed_data, d_out_vector, train_tensor.get_dev_pointer(), input_shape.size(), batch_size);
		}

		void add_function::back_propagate()
		{
			//copy the input derivatives to the output (as they are unchanged)
			cuda_safe_call(cudaMemcpy(d_out_derivatives, d_partial_derivatives, sizeof(float) * input_shape.size() * batch_size, cudaMemcpyDeviceToDevice));

			//if the function isn't locked calculate the partial derivativess
			if (!locked())
				avg_partial_derivatives();
		}

		void add_function::initialise(size_t batch_size)
		{
			//initialise the base function
			trainable_function::initialise(batch_size);

			//allocate relevant memory pointers for the variables required
			allocate_device_float_pointer(&d_der_vector, batch_size * input_shape.size());
			allocate_device_float_pointer(&d_pder_vector, batch_size * input_shape.size());
			allocate_device_float_pointer(&d_derivatives, batch_size * input_shape.size());

			//set the derivative vectors to 1
			fill_device_array(d_der_vector, 1, batch_size * input_shape.size());
			fill_device_array(d_pder_vector, 1, batch_size * input_shape.size());
		}

		void add_function::uninitialise() {
			//uninitialise the base function
			trainable_function::uninitialise();

			//deallocate all the memory pointers as they are finished with
			deallocate_device_float_pointer(d_der_vector);
			deallocate_device_float_pointer(d_derivatives);
		}

		void add_function::avg_partial_derivatives()
		{
			//average all the partial derivatives and write them to the derivative tensor
			average_vector(d_partial_derivatives, derivatives.get_dev_pointer(), input_shape.size(), batch_size, 1);
		}

		void add_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an ADD function
			__serialise(stream_buffer, offset, function_id::ADD);
		}

		matmul_function::matmul_function(size_t weight_rows, size_t weight_cols)
			: trainable_function(tensor({ weight_rows, weight_cols }))
		{
			//setup the input and output shape to the cols and rows respectively
			this->input_shape = shape(weight_cols);
			this->output_shape = shape(weight_rows);
		}

		matmul_function::matmul_function(tensor weights)
			: trainable_function(tensor(weights))
		{
			//setup the input and output shape to the tensor cols and rows respectively
			this->input_shape = shape(weights.get_shape()[1]);
			this->output_shape = shape(weights.get_shape()[0]);
		}

		matmul_function::~matmul_function()
		{
			//destroy the base function
			instruction_function::~instruction_function();
		}

		void matmul_function::run()
		{
			//make a placeholder matrix to pass into the multiply function
			d_matrix<float> B(batch_size, input_shape.width, feed_data);

			//multiply the weight matrix by the input matrix and write into the output matrix
			matrix_multiply<float, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL>(
				d_mat,
				B,
				d_out_vec
			);

			//ff we are training
			if (phase == PHASE_TRAINING) {
				//cache the input at this point into the partial derivative vector as it will be
				//needed for back propagation
				copy_into_device_array(feed_data, d_pder_vector, input_shape.width * batch_size, 0);
			}
		}

		void matmul_function::back_propagate()
		{
			//fill the out vector with 0s before multiplying
			cuda_safe_call(cudaMemset(d_out_derivatives, 0, sizeof(float) * input_shape.width * batch_size));

			//create placeholder matrices for multiplying on the device
			d_matrix<float> A({ output_shape.width, input_shape.width, get_train_vector() });
			d_matrix<float> B({ batch_size, output_shape.width, d_partial_derivatives });
			d_matrix<float> C({ batch_size, input_shape.width, d_out_derivatives });

			//multiply the two matrices A and B and write the product to C
			matrix_multiply<float, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL>(
				A,
				B,
				C
				);

			//if the function isn't locked calculate the partial derivativess
			if (!locked())
				avg_partial_derivatives();
		}

		void matmul_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			trainable_function::initialise(batch_size);

			//allocate the relevant pointers needed for this function
			allocate_device_float_pointer(&d_pder_vector, input_shape.width * batch_size);

			//set the derivative vector equal to the train tensor's pointer
			d_der_vector = train_tensor.get_dev_pointer();

			//setup device placeholder matrices for multiplication
			create_device_matrix(d_mat, input_shape.width, output_shape.width, train_tensor.get_dev_pointer());
			create_device_matrix(d_out_vec, batch_size, output_shape.width, d_out_vector);
		}

		void matmul_function::uninitialise()
		{
			//uninitialise the base function
			trainable_function::uninitialise();
		}

		void matmul_function::avg_partial_derivatives()
		{
			//create placeholder matrices for multiplying on the device
			d_matrix<float> A({ batch_size, output_shape.width, d_partial_derivatives });
			d_matrix<float> B({ input_shape.width, batch_size, d_pder_vector });
			d_matrix<float> C({ input_shape.width, output_shape.width, derivatives.get_dev_pointer() });

			//multiply the two matrices A and B and write the product to C
			matrix_multiply<float, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_ROW>(
				A,
				B,
				C
				);
		}

		void matmul_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an MATMUL function
			__serialise(stream_buffer, offset, function_id::MATMUL);
		}

		relu_function::relu_function(shape input_shape)
			: instruction_function()
		{
			//set the input and output shapes to the same
			this->input_shape = input_shape;
			this->output_shape = input_shape;
		}

		relu_function::~relu_function()
		{
			//destroy the base function
			instruction_function::~instruction_function();
		}

		void relu_function::run()
		{
			//apply elementwise ReLU activation over the input vector
			apply_relu(feed_data, d_out_vector, input_shape.size() * batch_size, 0);

			//if we are training
			if (phase == PHASE_TRAINING) {
				//calculate the derivatives vector from the inputs
				relu_derivative(feed_data, d_der_vector, input_shape.size() * batch_size, 0);
			}
		}

		void relu_function::back_propagate()
		{
			//multiply the current partial derivatives with the calculated derivative vectors elementwise
			hadamard_product(d_partial_derivatives, d_der_vector, d_out_derivatives, input_shape.size() * batch_size);
		}

		void relu_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//allocate the derivative pointer to the correct size
			allocate_device_float_pointer(&d_der_vector, input_shape.size() * batch_size);
		}

		void relu_function::uninitialise()
		{
			//uninitialise the base function
			instruction_function::uninitialise();

			//destroy the derivative vector pointer
			deallocate_device_float_pointer(d_der_vector);
		}

		void relu_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an RELU function
			__serialise(stream_buffer, offset, function_id::RELU);
		}

		leaky_relu_function::leaky_relu_function(shape input_shape, float alpha)
			: instruction_function()
		{
			//setup the input and output shapes to be the same and store the alpha constant
			this->input_shape = input_shape;
			this->output_shape = input_shape;
			this->alpha = alpha;
		}

		leaky_relu_function::~leaky_relu_function()
		{
			//apply elementwise leaky ReLU activation over the input vector
			instruction_function::~instruction_function();
		}

		void leaky_relu_function::run()
		{
			//apply elementwise leaky relu over the input elementwise
			apply_relu(feed_data, d_out_vector, input_shape.size() * batch_size, alpha);

			//if we are training
			if (phase == PHASE_TRAINING) {
				//calculate the derivatives vector from the inputs
				relu_derivative(feed_data, d_der_vector, input_shape.size() * batch_size, alpha);
			}
		}

		void leaky_relu_function::back_propagate()
		{
			//multiply the current partial derivatives with the calculated derivative vectors elementwise
			hadamard_product(d_partial_derivatives, d_der_vector, d_out_derivatives, input_shape.size() * batch_size);
		}

		void leaky_relu_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//allocate the derivative pointer to the correct size
			allocate_device_float_pointer(&d_der_vector, input_shape.size() * batch_size);
		}

		void leaky_relu_function::uninitialise()
		{
			//uninitialise the base function
			instruction_function::uninitialise();

			//destroy the derivative vector pointer
			deallocate_device_float_pointer(d_der_vector);
		}

		size_t leaky_relu_function::get_serialise_size()
		{
			//return the base size plus the size of 1 float (to store the alpha value)
			return instruction_function::get_serialise_size() + sizeof(float);
		}

		void leaky_relu_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an L_RELU function
			__serialise(stream_buffer, offset, function_id::L_RELU);
			size_t new_offset = instruction_function::get_serialise_size();

			//write the alpha value to the end of the buffer
			memcpy(&stream_buffer[new_offset], reinterpret_cast<char*>(reinterpret_cast<void*>(&alpha)), sizeof(float));
		}

		void leaky_relu_function::deserialise(char* stream_buffer, size_t offset)
		{
			//deserialise the base function from the buffer
			instruction_function::deserialise(stream_buffer, offset);
			size_t new_offset = instruction_function::get_serialise_size();

			//get the alpha value from the buffer
			memcpy(&alpha, reinterpret_cast<float*>(reinterpret_cast<void*>(&stream_buffer[new_offset])), sizeof(float));
		}

		tanh_function::tanh_function(shape input_shape)
		{
			//set the input and output shapes to the same
			this->input_shape = input_shape;
			this->output_shape = input_shape;
		}

		tanh_function::~tanh_function()
		{
			//destroy the base function
			instruction_function::~instruction_function();
		}

		void tanh_function::run()
		{
			//apply elementwise Tanh activation over the input vector
			apply_tanh(feed_data, d_out_vector, input_shape.size() * batch_size);

			//if we are training
			if (phase == PHASE_TRAINING) {
				//calculate the derivatives vector from the inputs
				tanh_derivative(feed_data, d_der_vector, input_shape.size() * batch_size);
			}
		}

		void tanh_function::back_propagate()
		{
			//multiply the current partial derivatives with the calculated derivative vectors elementwise
			hadamard_product(d_partial_derivatives, d_der_vector, d_out_derivatives, input_shape.size() * batch_size);
		}

		void tanh_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//allocate the derivative pointer to the correct size
			allocate_device_float_pointer(&d_der_vector, input_shape.size() * batch_size);
		}

		void tanh_function::uninitialise()
		{
			//uninitialise the base function
			instruction_function::uninitialise();

			//destroy the derivative vector pointer
			deallocate_device_float_pointer(d_der_vector);
		}

		void tanh_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an TANH function
			__serialise(stream_buffer, offset, function_id::TANH);
		}

		sigmoid_function::sigmoid_function(shape input_shape)
		{
			//set the input and output shapes to the same
			this->input_shape = input_shape;
			this->output_shape = input_shape;
		}

		sigmoid_function::~sigmoid_function()
		{
			//destroy the base function
			instruction_function::~instruction_function();
		}

		void sigmoid_function::run()
		{
			//apply elementwise Sigmoid activation over the input vector
			apply_sigmoid(feed_data, d_out_vector, input_shape.size() * batch_size);

			//if we are training
			if (phase == PHASE_TRAINING) {
				//calculate the derivatives vector from the inputs
				sigmoid_derivative(feed_data, d_der_vector, input_shape.size() * batch_size);
			}
		}

		void sigmoid_function::back_propagate()
		{
			//multiply the current partial derivatives with the calculated derivative vectors elementwise
			hadamard_product(d_partial_derivatives, d_der_vector, d_out_derivatives, input_shape.size() * batch_size);
		}

		void sigmoid_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//allocate the derivative pointer to the correct size
			allocate_device_float_pointer(&d_der_vector, input_shape.size() * batch_size);
		}

		void sigmoid_function::uninitialise()
		{
			//uninitialise the base function
			instruction_function::uninitialise();

			//destroy the derivative vector pointer
			deallocate_device_float_pointer(d_der_vector);
		}

		void sigmoid_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an SIGMOID function
			__serialise(stream_buffer, offset, function_id::SIGMOID);
		}

		//NOT IMPLEMENTED
		batch_normalisation_function::batch_normalisation_function(size_t input_size)
		{
			this->input_shape = { input_size, 1 };
		}

		//NOT IMPLEMENTED
		void batch_normalisation_function::run()
		{
		}

		//NOT IMPLEMENTED
		void batch_normalisation_function::back_propagate()
		{
		}

		//NOT IMPLEMENTED
		void batch_normalisation_function::initialise(size_t batch_size)
		{
			instruction_function::initialise(batch_size);
		}

		//NOT IMPLEMENTED
		void batch_normalisation_function::uninitialise()
		{
			instruction_function::uninitialise();
		}

		//NOT IMPLEMENTED
		void batch_normalisation_function::serialise(char* stream_buffer, size_t offset)
		{
			__serialise(stream_buffer, offset, function_id::BATCH_NORM);
		}

		conv2d_function::conv2d_function(shape input_shape, shape filter_shape, size_t n_filters, shape padding)
			: trainable_function(tensor({ filter_shape.width, filter_shape.height, filter_shape.depth, n_filters }))
		{
			//setup shapes from input parameters
			this->filter_shape = filter_shape;
			this->output_shape.depth = n_filters;
			this->padding = padding;

			//set the input shape using the macro
			set_input_shape(input_shape);
		}

		conv2d_function::conv2d_function(tensor filter, shape padding)
			: trainable_function(filter)
		{
			//check that the filter has 4 dimensions (Width, Height, Depth, Num of Filters)
			ERR_ASSERT(filter.get_dimensions() != 4, "Conv2d filter must be four dimensional (width, height, depth, filters)");

			//set the filter shape to the first 3 dimensions
			this->filter_shape = shape(filter.get_shape()[0], filter.get_shape()[1], filter.get_shape()[2]);

			//set the output shape depth to the number of filters
			this->output_shape.depth = filter.get_shape()[3];

			//set the local padding variable
			this->padding = padding;
		}

		conv2d_function::~conv2d_function()
		{
			//destroy the base function
			trainable_function::~trainable_function();
		}

		void conv2d_function::run()
		{
			cuda_safe_call(cudaMemset(d_out_vector, 0, sizeof(float) * output_shape.size() * batch_size));

			//convolve the filter over the input space to give an output map
			filter_convolve_2d(
				feed_data,
				get_filter().get_dev_pointer(),
				d_out_vector,
				input_shape,
				output_shape,
				filter_shape,
				padding,
				batch_size
			);

			if (phase == PHASE_TRAINING) {
				//cache the input ready for back propagation
				cuda_safe_call(cudaMemcpy(d_pder_vector, feed_data, sizeof(float) * input_shape.size() * batch_size, cudaMemcpyDeviceToDevice));
			}
		}

		void conv2d_function::back_propagate()
		{
			//set the output array to all 0s
			cuda_safe_call(cudaMemset(d_out_derivatives, 0, sizeof(float) * input_shape.size() * batch_size));

			//perform outer convolution between the input and the filter
			filter_outer_convolve_2d(
				d_partial_derivatives,
				get_filter().get_dev_pointer(),
				d_out_derivatives,
				output_shape,
				input_shape,
				filter_shape,
				padding,
				batch_size
			);

			//if the function isn't locked calculate the partial derivativess
			if (!locked())
				avg_partial_derivatives();
		}

		void conv2d_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			trainable_function::initialise(batch_size);

			//allocate the relevant pointers needed for this function
			allocate_device_float_pointer(&d_pder_vector, input_shape.size() * batch_size);
		}

		void conv2d_function::uninitialise()
		{
			//uninitialise the base function
			trainable_function::uninitialise();
		}

		void conv2d_function::avg_partial_derivatives()
		{
			//reset the derivatives tensor vector to 0s ready for calculating the derivatives
			cuda_safe_call(cudaMemset(derivatives.get_dev_pointer(), 0, sizeof(float) * filter_shape.size() * output_shape.depth));

			//calculate the derivatives with respect to each filter variable
			filter_convolve_2d_derivative(
				feed_data,
				d_partial_derivatives,
				derivatives.get_dev_pointer(),
				input_shape,
				output_shape,
				filter_shape,
				padding,
				batch_size
			);
		}

		size_t conv2d_function::get_serialise_size()
		{
			//return the base size plus the size of 1 shape (to store the padding)
			return trainable_function::get_serialise_size() + sizeof(shape);
		}

		void conv2d_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an CONV_2D function
			__serialise(stream_buffer, offset, function_id::CONV_2D);
			size_t new_offset = trainable_function::get_serialise_size() + offset;

			//write the padding to the end of the stream
			memcpy(&stream_buffer[new_offset], padding.serialise(), sizeof(shape));
		}

		void conv2d_function::deserialise(char* stream_buffer, size_t offset)
		{
			//deserialise the base function from the buffer
			trainable_function::deserialise(stream_buffer, offset);

			//retrieve the filter shape from the deserialised training tensor
			vector<size_t> t_shape = train_tensor.get_shape();
			filter_shape = shape(t_shape[0], t_shape[1], t_shape[2]);
			size_t new_offset = trainable_function::get_serialise_size() + offset;

			//deserialise the padding from the buffer
			padding.deserialise(stream_buffer, new_offset);
		}

		void conv2d_function::set_input_shape(shape input_shape)
		{
			//check that the input depth and filter depth are the same
			ERR_ASSERT(input_shape.depth != filter_shape.depth, "Input depth and filter depth must be equal");

			//set the input shapes equal
			this->input_shape = input_shape;

			//setup the offset output shape relative to padding and the filter size
			this->output_shape.width = input_shape.width - filter_shape.width + 1 + padding.width * 2;
			this->output_shape.height = input_shape.height - filter_shape.height + 1 + padding.height * 2;
		}

		max_pool_function::max_pool_function(shape pool_size, shape stride)
		{
			//set the pool size and stride from inputs
			this->pool_size = pool_size;
			this->stride = stride;
		}

		max_pool_function::~max_pool_function()
		{
			//destroy the base function
			instruction_function::~instruction_function();
		}

		void max_pool_function::run()
		{
			//perform max pooling over the input space with the pool size and stride specified
			max_pool_2d(feed_data, d_mask, d_out_vector, input_shape, pool_size, stride, output_shape, padding, batch_size);
		}

		void max_pool_function::back_propagate()
		{
			//initialise the derivative vector to 0s
			cuda_safe_call(cudaMemset(d_out_derivatives, 0, sizeof(float) * input_shape.size() * batch_size));

			//perform max pooing derivative over the inputs to get the partial derivatives with respect
			//to the inputs
			max_pool_2d_derivative(d_partial_derivatives, d_mask, d_out_derivatives, output_shape, input_shape, batch_size);
		}

		void max_pool_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//allocate an int pointer for the mask (as it holds indices)
			cuda_safe_call(cudaMallocManaged(&d_mask, sizeof(int) * input_shape.size() * batch_size));

			//allocate the derivative placeholder vector
			allocate_device_float_pointer(&d_der_vector, input_shape.size() * batch_size);
		}

		void max_pool_function::uninitialise()
		{
			//uninitialise the base function
			instruction_function::uninitialise();

			//destroy pointers now that they are finished with
			cuda_safe_call(cudaFree(d_mask));
			deallocate_device_float_pointer(d_der_vector);
		}

		void max_pool_function::set_input_shape(shape input_shape)
		{
			//set the padding equal to the remainder after striding
			padding.width = (input_shape.width - pool_size.width) % stride.width;
			padding.height = (input_shape.height - pool_size.height) % stride.height;

			//set the input shape from the parameters
			this->input_shape = input_shape;

			//calculate the output shape from the input shape, pool size and stride
			this->output_shape = shape(
				(input_shape.width + padding.width) / stride.width,
				(input_shape.height + padding.height) / stride.height,
				input_shape.depth
			);
		}

		size_t max_pool_function::get_serialise_size()
		{
			//return the base size plus the size of 2 shapes (to store the pool size and stride)
			return instruction_function::get_serialise_size() + sizeof(shape) * 2;
		}

		void max_pool_function::serialise(char* stream_buffer, size_t offset)
		{
			//serialise the function flagging that this is an POOL function
			__serialise(stream_buffer, offset, function_id::POOL);
			size_t new_offset = instruction_function::get_serialise_size() + offset;

			//write the pool size and stride to the end of the stream
			memcpy(&stream_buffer[new_offset], pool_size.serialise(), sizeof(shape));
			memcpy(&stream_buffer[new_offset + sizeof(shape)], stride.serialise(), sizeof(shape));
		}

		void max_pool_function::deserialise(char* stream_buffer, size_t offset)
		{
			//deserialise the base function from the buffer
			instruction_function::deserialise(stream_buffer, offset);
			size_t new_offset = instruction_function::get_serialise_size() + offset;

			//deserialise the pool size and stride from the buffer
			pool_size.deserialise(stream_buffer, new_offset);
			stride.deserialise(stream_buffer, new_offset + sizeof(shape));

			//set the padding equal to the remainder after striding
			padding.width = (input_shape.width - pool_size.width) % stride.width;
			padding.height = (input_shape.height - pool_size.height) % stride.height;
		}

		void dropout_function::run()
		{
			if (phase == PHASE_TRAINING) {
				//get an array of random numbers for which nodes to keep
				random_dropout_array(dropout_mask, keep_rate, batch_size * input_shape.size());

				//multiply the dropped nodes by 0 to remove them
				hadamard_product(feed_data, dropout_mask, d_out_vector, batch_size * input_shape.size());
			}
		}

		void dropout_function::back_propagate()
		{
			//get the partial derivatives with respect to the inputs
			hadamard_product(d_partial_derivatives, dropout_mask, d_out_derivatives, batch_size * input_shape.size());
		}

		void dropout_function::initialise(size_t batch_size)
		{
			//call the base initialiser
			instruction_function::initialise(batch_size);

			//allocate the mask pointer
			allocate_device_float_pointer(&dropout_mask, batch_size * input_shape.size());
		}

		void dropout_function::uninitialise()
		{
			//uninitialise the base function
			instruction_function::uninitialise();

			//destroy pointers now that they are finished with
			deallocate_device_float_pointer(dropout_mask);
		}
	}
}
