#include "stdafx.h"

#include "network_model.h"


namespace nnet {
	void network_model::entry(shape entry_shape)
	{
		//set the current layer shape to the entry shape
		this->init_layer_shape = entry_shape;

		//flag that an entry was specified
		__ent_spec = true;
	}

	add_function* network_model::add(tensor biases)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create an add function
		add_function* f = new add_function(biases);

		//push the function into the instruction function list
		instructions.push_back(f);

		//push it to trainable functions as add functions are trainable
		train_funcs.push_back(f);

		return f;
	}

	matmul_function* network_model::matmul(tensor weights)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//check the tensor is a matrix
		if (weights.get_dimensions() != 2)
			throw exception("Weight tensor must be two dimensional");

		//create a matmul function
		matmul_function* f = new matmul_function(weights);

		//push the function into the instruction function list
		instructions.push_back(f);

		//push it to trainable functions as add functions are trainable
		train_funcs.push_back(f);

		//set the shape of the next layer to be the output shape from the matrix
		this->init_layer_shape = shape(weights.get_shape()[0]);

		return f;
	}

	dense_layer network_model::dense(size_t units, variable_initialiser weight_initialiser, variable_initialiser bias_initialiser) {
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create a weight tensor with random sampling from a normal distribution
		tensor weights = tensor::random_normal({ units, init_layer_shape.width }, weight_initialiser.mean, weight_initialiser.stddev);

		//create a bias tensor with random sampling from a normal distribution
		tensor biases = tensor::random_normal(units, weight_initialiser.mean, weight_initialiser.stddev);

		//add a matmul layer
		matmul_function* m = matmul(weights);

		//add a bias layer
		add_function* a = add(biases);

		//set the next layer shape to be the output shape
		init_layer_shape = shape(units);

		return dense_layer(a, m);
	}

	conv2d_function* network_model::conv2d(shape filter_shape, size_t n_filters, padding_type padding, variable_initialiser initialiser)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		switch (padding) {
		case padding_type::PADDING_VALID:
			//if the padding is valid, create a conv2d with no padding
			return conv2d(filter_shape, n_filters, shape(0, 0), initialiser);
		case padding_type::PADDING_SAME:
			//if the padding is same, work out the padding size
			return conv2d(filter_shape, n_filters, shape((filter_shape.width - 1) / 2, (filter_shape.height - 1) / 2), initialiser);
		}
	}

	conv2d_function* network_model::conv2d(shape filter_shape, size_t n_filters, shape padding, variable_initialiser initialiser)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//if the filter_shape depth is unset (so 1) work it out from the filter depth
		size_t f_depth = filter_shape.depth;
		if (filter_shape.depth == 1)
			f_depth = init_layer_shape.depth;

		//create the filter tensor from a normal distribution
		tensor filter = tensor::random_normal({ filter_shape.width, filter_shape.height, f_depth, n_filters }, initialiser.mean, initialiser.stddev);

		//call the next level
		return conv2d(filter, padding);
	}

	conv2d_function* network_model::conv2d(tensor filter, shape padding)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//check the filter has the right number of dimensions
		if (filter.get_dimensions() != 4)
			throw exception("Conv2d filter must be four dimensional (width, height, depth, filters)");

		//create the conv2d function
		conv2d_function * f = new conv2d_function(filter, padding);

		//push the function into the instruction function list
		instructions.push_back(f);

		//push it to trainable functions as add functions are trainable
		train_funcs.push_back(f);

		//set the input shape of the function, which for conv2d will calculate
		//the output shape
		f->set_input_shape(init_layer_shape);
		
		//set the next layer shape to be the output shape
		init_layer_shape = f->output_shape;

		return f;
	}

	max_pool_function* network_model::max_pool(shape pool_size, shape stride)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the max pool function
		max_pool_function * f = new max_pool_function(pool_size, stride);

		//push the function into the instruction function list
		instructions.push_back(f);

		//set the input shape of the function, which for max pool will calculate
		//the output shape
		f->set_input_shape(init_layer_shape);

		//set the next layer shape to be the output shape
		init_layer_shape = f->output_shape;

		return f;
	}

	flatten_function* network_model::flatten()
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the flatten function 
		flatten_function * f = new flatten_function(init_layer_shape);

		//push the function into the instruction function list
		instructions.push_back(f);

		//set the next layer shape to be the output shape
		init_layer_shape = f->output_shape;

		return f;
	}

	reshape_function* network_model::reshape(shape output_shape)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the reshape function
		reshape_function * f = new reshape_function(init_layer_shape, output_shape);

		//push the function into the instruction function list
		instructions.push_back(f);

		//set the next layer shape to be the output shape
		init_layer_shape = output_shape;

		return f;
	}

	dropout_function* network_model::dropout(float keep_rate)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the reshape function
		dropout_function* f = new dropout_function(keep_rate);

		f->set_input_shape(init_layer_shape);

		//push the function into the instruction function list
		instructions.push_back(f);

		return f;
	}

	relu_function* network_model::relu()
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the relu function 
		relu_function* f = new relu_function(init_layer_shape);

		//push the function into the instruction function list
		instructions.push_back(f);

		return f;
	}

	leaky_relu_function* network_model::leaky_relu(float alpha)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");
		
		//create the leaky relu function 
		leaky_relu_function* f = new leaky_relu_function(init_layer_shape, alpha);

		//push the function into the instruction function list
		instructions.push_back(f);

		return f;
	}

	tanh_function* network_model::tanh()
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the tanh function 
		tanh_function* f = new tanh_function(init_layer_shape);

		//push the function into the instruction function list
		instructions.push_back(f);

		return f;
	}

	sigmoid_function* network_model::sigmoid()
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//create the sigmoid function 
		sigmoid_function* f = new sigmoid_function(init_layer_shape);

		//push the function into the instruction function list
		instructions.push_back(f);

		return f;
	}

	instruction_function* network_model::function(instruction_function *func)
	{
		//check that the entry shape is specified
		if (!__ent_spec)
			throw exception("Entry size not specified");

		//add the function passed in the the instructions
		instructions.push_back(func);

		return func;
	}
	
	void network_model::add_logger(analytics logger)
	{
		analytics_logger = &logger;
	}
	
	void network_model::initialise_model(size_t batch_size)
	{
		//if the model is already initialised abort and return
		if (model_initialised)
			return;

		//check that there is a model to initialise
		if (instructions.size() == 0)
			throw std::exception("Cannot initialise empty model");

		//initialise the largest layer size to 0
		largest_layer_size = 0;

		//add the starting shape to the layer sizes
		layer_shapes.push_back(instructions[0]->input_shape);

		for (int i = 0; i < instructions.size(); i++) {
			//initialise the function
			instructions[i]->initialise(batch_size);

			//add the function shape to the layer_shapes vector
			layer_shapes.push_back(instructions[i]->output_shape);

			//if the current size is greater than the current largest, update the current largest
			if (instructions[i]->input_shape.size() > largest_layer_size)
				largest_layer_size = instructions[i]->input_shape.size();
		}

		//get a reference to the output shape from the model
		output_shape = instructions.back()->output_shape;

		//if there is a cost function, initialise it
		if (cost_func != nullptr)
			cost_func->initialise(output_shape, batch_size, -1);

		//if there is an output function, initialise it
		if (output_func != nullptr)
			output_func->initialise(output_shape, batch_size);

		//if there is an optimiser, initialise it
		if (opt != nullptr)
			opt->initialise(train_funcs);

		//save the batch size
		this->batch_size = batch_size;

		//flag that the model is now initialised
		model_initialised = true;
	}

	void network_model::uninitialise_model()
	{
		//if the model was never initialised, don't try to uninitialise and return
		if (!model_initialised)
			return;

		//uninitalise each function in the model
		for (int i = 0; i < instructions.size(); i++) {
			instructions[i]->uninitialise();
		}

		//if there was a cost function, uninitialise it
		if (cost_func != nullptr)
			cost_func->uninitialise();

		//if there was a output function, uninitialise it
		if (output_func != nullptr)
			output_func->uninitialise();

		//if there was a optimiser, uninitialise it
		if (opt != nullptr)
			opt->uninitialise();

		//flag that the model is now uninitialised
		model_initialised = false;
	}
	
	tensor network_model::run(tensor input)
	{
		//if the model is not initialised it can not be run, so throw an exception
		if (!model_initialised)
			throw exception("Model not initialised");

		//initialise the input tensor in case it has not been initialised
		input.initialise();

		//declare pointers for batches, layers and outputs
		float * d_in_batch, *d_in_layer, * d_out_layer, * d_output;

		//falsely initialse the d_out_layer to avoid compiler errors
		d_out_layer = new float();

		//get the number of inputs, which should be in the first dimension of the input tensor
		size_t num = input.get_shape()[0];

		//get the shape of the input to the model
		shape input_shape = layer_shapes[0];

		//allocate a placeholder pointer for the output of each batch
		allocate_device_float_pointer(&d_output, num * output_shape.size());

		//calculate the number of batches depending on the number of inputs and the batch size
		int n_batches = ceil(num / (float)batch_size);

		//begin with the current batch size being the primary batch size
		size_t current_batch_size = batch_size;

		//loop through each batch
		for (int batch = 0; batch < n_batches; batch++) {

			//if this is the last batch, reduce the current batch size (if necessary)
			if (batch == n_batches - 1) {
				current_batch_size = num % batch_size;
				if (current_batch_size == 0)
					current_batch_size = batch_size;
			}

			//set the placeholder pointer to be the input batch
			d_in_batch = &input.get_dev_pointer()[batch * batch_size * input_shape.size()];

			//set the first input layer to be the input
			d_in_layer = d_in_batch;

			//loop through each function in the model
			for (int i = 0; i < instructions.size(); i++) {
				//if this is a train only function, skip over it
				if (instructions[i]->get_type() & instruction_function_type::TRAIN_ONLY)
					continue;

				//get a reference to the output from this layer
				d_out_layer = instructions[i]->get_out_vector();

				//reset the output vector to 0s for this layer
				fill_device_array(d_out_layer, 0, instructions[i]->output_shape.size());

				//run this layer with the given input
				instructions[i]->run(d_in_layer, current_batch_size);

				//set the next input to the output from this layer
				d_in_layer = d_out_layer;
			}

			//if there is an output function, run the output from the instructions through this function
			//and set the output to this new vector
			if (output_func != nullptr) {
				output_func->run(d_out_layer, current_batch_size);
				d_out_layer = output_func->get_out_vector();
			}

			//copy the batch output into the full output
			copy_into_device_array(d_out_layer, d_output, current_batch_size * output_shape.size(), batch * batch_size * output_shape.size());
		}

		//allocate a host float pointer for the results
		float * output = (float *)malloc(sizeof(float) * num * output_shape.size());

		//copy the results to the host
		retrieve_output_data(output, d_output, output_shape.size() * num);

		//destroy the placeholder pointer
		deallocate_device_float_pointer(d_output);

		//declare a tensor for the result
		tensor * ret_tensor;

		//create the result tensor from the output 
		if (output_shape.height == 1)
			ret_tensor = new tensor({ num, output_shape.width }, output);
		else
			ret_tensor = new tensor({ num, output_shape.width, output_shape.height }, output);

		//return the tensor stored at the pointer location
		return *ret_tensor;
	}
	
	void network_model::train(tensor train_x, tensor train_y, int epochs)
	{
		//if the model is not initialised it can not be run, so throw an exception
		if (!model_initialised)
			throw exception("Model not initialised");

		//initialise the input tensors in case they have not been initialised
		train_x.initialise();
		train_y.initialise();

		//get the number of inputs, which should be in the first dimension of the input tensor
		size_t num = train_x.get_shape()[0];

		//get the shape of the input to the model
		shape input_shape = layer_shapes[0];

		//calculate the number of batches depending on the number of inputs and the batch size
		int n_batches = ceil(num / (float)batch_size);

		//begin with the current batch size being the primary batch size
		size_t current_batch_size = batch_size;

		//if there is a logger on the model, begin logging
		if (analytics_logger != nullptr) {
			analytics_logger->init_logging();
		}

		//loop through each epoch. Every epoch represents training against the entire tensors
		//specified
		for (int epoch = 0; epoch < epochs; epoch++) {
			//if there is a logger, call its "on epoch start" event, as this epoch is
			//starting
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_start();
			}

			//clear the loss metric in the cost function
			cost_func->clear_loss();

			//set the total loss for this epoch to 0
			float epoch_loss = 0;

			//set the current batch size to the primary batch size
			current_batch_size = batch_size;

			//loop through each batch
			for (int batch = 0; batch < n_batches; batch++) {
				//if there is a logger, call the "on step start" event, as this
				//step is starting
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_start();
				}

				//if this is the last batch, reduce the current batch size (if necessary)
				if (batch == n_batches - 1) {
					current_batch_size = num % batch_size;
					if (current_batch_size == 0)
						current_batch_size = batch_size;
				}

				//calculate the gradients for this entire batch with the batch read from the input tensors
				calc_batch_gradient(
					&train_x.get_dev_pointer()[batch * batch_size * input_shape.size()],
					&train_y.get_dev_pointer()[batch * batch_size * output_shape.size()],
					current_batch_size
				);

				//optimise the trainable functions in the network with the optimiser
				opt->optimise();

				//add the total loss for this batch to the epoch loss
				epoch_loss += cost_func->get_average_loss() * current_batch_size;

				//if there is a logger, call the "on step end" event function, as this step
				//has just ended
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_end(cost_func->get_average_loss());
				}
			}

			//if there is a logger, call the "on epoch end" event function, as this epoch has just
			//ended
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_end(epoch_loss / num);
			}
		}

		//if there is a logger, call the end_logging() function to stop logging
		if (analytics_logger != nullptr) {
			analytics_logger->end_logging();
		}
	}

	void network_model::train(batch_iterator & b_iter, int epochs)
	{
		//if the model is not initialised it can not be run, so throw an exception
		if (!model_initialised)
			throw exception("Model not initialised");

		//initialise the iterator with the same batch size as the model
		b_iter.initialise(batch_size);

		//get the number of inputs, which is known by the iterator
		size_t num = b_iter.get_size();

		//calculate the number of batches depending on the number of inputs and the batch size
		int n_batches = ceil(num / (float)batch_size);

		//begin with the current batch size being the primary batch size
		size_t current_batch_size = batch_size;

		//if there is a logger on the model, begin logging
		if (analytics_logger != nullptr) {
			analytics_logger->init_logging();
		}

		//loop through each epoch. Every epoch represents training against the entire iterator
		//specified
		for (int epoch = 0; epoch < epochs; epoch++) {
			//if there is a logger, call its "on epoch start" event, as this epoch is
			//starting
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_start();
			}

			//clear the loss metric in the cost function
			cost_func->clear_loss();

			//set the total loss for this epoch to 0
			float epoch_loss = 0;

			//set the current batch size to the primary batch size
			current_batch_size = batch_size;

			//loop through each batch
			for (int batch = 0; batch < n_batches; batch++) {
				//if there is a logger, call the "on step start" event, as this
				//step is starting
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_start();
				}

				//if this is the last batch, reduce the current batch size (if necessary)
				if (batch == n_batches - 1) {
					current_batch_size = num % batch_size;
					if (current_batch_size == 0)
						current_batch_size = batch_size;
				}

				//load the next batch into the iterator
				b_iter.next_batch();

				//get the next batch data
				tensor * train_x = b_iter.get_next_batch();

				//get the next batch labels
				tensor * train_y = b_iter.get_next_batch_labels();

				//initialise these tensors in case they have not been initialised by the batch
				//iterator (note: all API iterators will already have initialsied these tensors)
				train_x->initialise();
				train_y->initialise();

				//calculate the gradients for this entire batch with the tensors loaded
				//from the iterator
				calc_batch_gradient(
					train_x->get_dev_pointer(),
					train_y->get_dev_pointer(),
					current_batch_size
				);

				//optimise the trainable functions in the network with the optimiser
				opt->optimise();

				//add the total loss for this batch to the epoch loss
				epoch_loss += cost_func->get_average_loss() * current_batch_size;

				//if there is a logger, call the "on step end" event function, as this step
				//has just ended
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_end(cost_func->get_average_loss());
				}
			}

			//reset the iterator back to the beginning ready for the next epoch
			b_iter.reset_iterator();

			//if there is a logger, call the "on epoch end" event function, as this epoch has just
			//ended
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_end(epoch_loss / num);
			}
		}

		//if there is a logger, call the end_logging() function to stop logging
		if (analytics_logger != nullptr) {
			analytics_logger->end_logging();
		}
	}

	float network_model::evaluate(tensor test_x, tensor test_y)	{
		//if the model is not initialised it can not be run, so throw an exception
		if (!model_initialised)
			throw exception("Model not initialised");

		//initialise the input tensors in case they have not been initialised
		test_x.initialise();
		test_y.initialise();

		//declare pointers for batches, layers and outputs
		float * d_in_batch, *d_in_layer, *d_out_layer;

		//falsely initialse the d_out_layer to avoid compiler errors
		d_out_layer = new float();

		//get the number of inputs, which should be in the first dimension of the input tensor
		size_t num = test_x.get_shape()[0];

		//get the shape of the input to the model
		shape input_shape = layer_shapes[0];

		//calculate the number of batches depending on the number of inputs and the batch size
		int n_batches = ceil(num / (float)batch_size);

		//begin with the current batch size being the primary batch size
		size_t current_batch_size = batch_size;

		//create a device int to hold the total number of correct results
		unsigned int * d_total_correct;

		//declare the correct pointer
		cuda_safe_call(cudaMallocManaged(&d_total_correct, sizeof(int)));

		//set the value to 0
		cuda_safe_call(cudaMemset(d_total_correct, 0, sizeof(int)));

		//loop through each batch
		for (int batch = 0; batch < n_batches; batch++) {
			//if there is a logger, call the "on step start" event, as this
			//step is starting
			if (batch == n_batches - 1) {
				current_batch_size = num % batch_size;
				if (current_batch_size == 0)
					current_batch_size = batch_size;
			}

			//set the placeholder pointer to be the input batch
			d_in_batch = &test_x.get_dev_pointer()[batch * batch_size * input_shape.size()];

			//set the first input layer to be the input
			d_in_layer = d_in_batch;

			//loop through each function in the model
			for (int i = 0; i < instructions.size(); i++) {
				//if this is a train only function, skip over it
				if (instructions[i]->get_type() & instruction_function_type::TRAIN_ONLY)
					continue;

				//get a reference to the output from this layer
				d_out_layer = instructions[i]->get_out_vector();

				//reset the output vector to 0s for this layer
				fill_device_array(d_out_layer, 0, batch_size * instructions[i]->output_shape.size());

				//run this layer with the given input
				instructions[i]->run(d_in_layer, current_batch_size);

				//set the next input to the output from this layer
				d_in_layer = d_out_layer;
			}

			//calculate no. correct values

			//declare a pointer for the argmax result of the output layer
			int * d_out_argmax;

			//allocate memory
			cuda_safe_call(cudaMallocManaged(&d_out_argmax, sizeof(int) * current_batch_size));

			//get the argmax for each output
			apply_argmax(d_out_layer, d_out_argmax, output_shape.size(), current_batch_size);

			//if the test has 1 dimension, it is sparse encoded, else it is one hot encoded
			if (test_y.get_dimensions() == 1) {
				//compare the test pointer directly to the output pointer and accumulate results
				comp_eq(d_out_argmax, test_y.get_dev_pointer(), d_total_correct, current_batch_size);
			}
			else if (test_y.get_dimensions() == 2) {
				//declare a pointer for the argmax of the one hot tensor
				int * d_label_argmax;

				//allocate memory for the argmax vector
				cuda_safe_call(cudaMallocManaged(&d_label_argmax, sizeof(int) * current_batch_size));

				//get the argmax for each label
				apply_argmax(test_y.get_dev_pointer(), d_label_argmax, output_shape.size(), current_batch_size);

				//compare the test pointer to the output pointer and accumulate results
				comp_eq(d_out_argmax, d_label_argmax, d_total_correct, current_batch_size);
			}
		}

		//int to hold the number of correctly classified stimuli
		int total_correct;

		cuda_safe_call(cudaMemcpy(&total_correct, d_total_correct, sizeof(int), cudaMemcpyDeviceToHost));

		//return the total correct divided by the total number to give the the percentage of
		//correctly classified elements
		return total_correct / (float)num;
	}

	float network_model::evaluate(batch_iterator & b_iter)
	{
		//if the model is not initialised it can not be run, so throw an exception
		if (!model_initialised)
			throw exception("Model not initialised");

		//initialise the iterator with the same batch size as the model
		b_iter.initialise(batch_size);

		//declare pointers for batches, layers and outputs
		float * d_in_batch, *d_in_layer, *d_out_layer;

		//falsely initialse the d_out_layer to avoid compiler errors
		d_out_layer = new float();

		//get the number of inputs, which is known by the iterator
		size_t num = b_iter.get_size();

		//calculate the number of batches depending on the number of inputs and the batch size
		int n_batches = ceil(num / (float)batch_size);

		//begin with the current batch size being the primary batch size
		size_t current_batch_size = batch_size;

		//create a device int to hold the total number of correct results
		unsigned int * d_total_correct;

		//declare the correct pointer
		cuda_safe_call(cudaMallocManaged(&d_total_correct, sizeof(int)));

		//set the value to 0
		cuda_safe_call(cudaMemset(d_total_correct, 0, sizeof(int)));

		//loop through each batch
		for (int batch = 0; batch < n_batches; batch++) {
			//if there is a logger, call the "on step start" event, as this
			//step is starting
			if (batch == n_batches - 1) {
				current_batch_size = num % batch_size;
				if (current_batch_size == 0)
					current_batch_size = batch_size;
			}

			//load the next batch into the iterator
			b_iter.next_batch();

			//get the next batch data
			d_in_batch = b_iter.get_next_batch()->get_dev_pointer();

			//set the first input layer to be the input
			d_in_layer = d_in_batch;

			//loop through each function in the model
			for (int i = 0; i < instructions.size(); i++) {
				//if this is a train only function, skip over it
				if (instructions[i]->get_type() & instruction_function_type::TRAIN_ONLY)
					continue;

				//get a reference to the output from this layer
				d_out_layer = instructions[i]->get_out_vector();

				//reset the output vector to 0s for this layer
				fill_device_array(d_out_layer, 0, batch_size * instructions[i]->output_shape.size());

				//run this layer with the given input
				instructions[i]->run(d_in_layer, current_batch_size);

				//set the next input to the output from this layer
				d_in_layer = d_out_layer;
			}

			//calculate no. correct values

			//declare a pointer for the argmax result of the output layer
			int * d_out_argmax;

			//allocate memory
			cuda_safe_call(cudaMallocManaged(&d_out_argmax, sizeof(int) * current_batch_size));

			//get the argmax for each output
			apply_argmax(d_out_layer, d_out_argmax, output_shape.size(), current_batch_size);

			tensor test_y = *b_iter.get_next_batch_labels();

			//if the test has 1 dimension, it is sparse encoded, else it is one hot encoded
			if (test_y.get_dimensions() == 1) {
				//compare the test pointer directly to the output pointer and accumulate results
				comp_eq(d_out_argmax, test_y.get_dev_pointer(), d_total_correct, current_batch_size);
			}
			else if (test_y.get_dimensions() == 2) {
				//declare a pointer for the argmax of the one hot tensor
				int * d_label_argmax;

				//allocate memory for the argmax vector
				cuda_safe_call(cudaMallocManaged(&d_label_argmax, sizeof(int) * current_batch_size));

				//get the argmax for each label
				apply_argmax(test_y.get_dev_pointer(), d_label_argmax, output_shape.size(), current_batch_size);

				//compare the test pointer to the output pointer and accumulate results
				comp_eq(d_out_argmax, d_label_argmax, d_total_correct, current_batch_size);
			}
		}

		//int to hold the number of correctly classified stimuli
		int total_correct;

		cuda_safe_call(cudaMemcpy(&total_correct, d_total_correct, sizeof(int), cudaMemcpyDeviceToHost));

		//return the total correct divided by the total number to give the the percentage of
		//correctly classified elements
		return total_correct / (float)num;
	}

	void network_model::write_model_to_file(string model_folder, string model_name)
	{
		//create the directory to save the model in (only works for Windows afaik)
		CreateDirectoryA((LPCSTR)(model_folder + "\\" + model_name).c_str(), NULL);

		//create a byte stream to write out to the .model file
		ofstream data_stream(model_folder + "\\" + model_name + "\\" + model_name + ".model", ofstream::binary);

		//get the number of instructions in the model
		int i_size = instructions.size();

		//write the number of instructions to the beginning of the file
		data_stream.write((char *)&i_size, sizeof(int));

		//loop through each layer in the model
		for (int layer = 0; layer < i_size; layer++) {
			//get a temporary reference to the layer function
			instruction_function * i_func = instructions[layer];

			//get the size which this function will stream to (in bytes)
			size_t stream_size = i_func->get_serialise_size();

			//allocate a buffer to serialise the function into
			char * stream_buffer = (char *)malloc(stream_size);

			//serialise the function into the beginning of this buffer
			i_func->serialise(stream_buffer, 0);

			//write the stream size for this function to the file, so when it is reloaded the reader knows how far
			//to read for this function
			data_stream.write(reinterpret_cast<char*>(reinterpret_cast<void*>(&stream_size)), sizeof(size_t));

			//write the serialised buffer to the file stream
			data_stream.write(stream_buffer, stream_size);
		}

		//close the file stream as we are now finished writing
		data_stream.close();
	}
	
	network_model network_model::load_model_from_file(string model_folder, string model_name)
	{
		//create a model instance
		network_model * model = new network_model();

		//open the data file
		ifstream data_stream(model_folder + "\\" + model_name + "\\" + model_name + ".model", ifstream::binary);

		//get the total length of the file
		data_stream.seekg(0, data_stream.end);
		size_t length = data_stream.tellg();
		data_stream.seekg(0, data_stream.beg);

		//create a buffer for the first int
		char n_layers_b[4];
		int n_layers;

		//read in the 4 byte buffer
		data_stream.read(n_layers_b, sizeof(int));

		//cast the 4 bytes to an integer, which was saved as the number of instructions
		//when writing the file
		n_layers = *(int *)n_layers_b;

		//loop through each layer (as we now know how many layers there are)
		for (int layer = 0; layer < n_layers; layer++) {
			//read in the size of this function so we know how far to read
			size_t layer_stream_size;
			data_stream.read((char *)&layer_stream_size, sizeof(size_t));

			//create a suitably sized data buffer for this layer
			char * data_buff = (char *)malloc(layer_stream_size);

			//read the layer into the placeholder buffer
			data_stream.read(data_buff, layer_stream_size);

			//the first part of the buffer should be the id of the function for deserialisation
			function_id func_id = *reinterpret_cast<function_id*>(reinterpret_cast<void*>(&data_buff[0]));

			//declare a pointer for the current function
			instruction_function * f;

			//switch the id and create the matching function
			//type and deserialise it to set it up from the buffer
			switch (func_id) {
			case function_id::ADD:
				f = new add_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::MATMUL:
				f = new matmul_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::RELU:
				f = new relu_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::L_RELU:
				f = new leaky_relu_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::TANH:
				f = new tanh_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::SIGMOID:
				f = new sigmoid_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::BATCH_NORM:
				break;
			case function_id::CONV_2D:
				f = new conv2d_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::POOL:
				f = new max_pool_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::RESHAPE:
				f = new reshape_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::FLATTEN:
				f = new flatten_function();
				f->deserialise(data_buff, 0);
				break;
			}

			//add the function to the instructions vector
			model->instructions.push_back(f);

			//if the function is alos trainable, add it to the train functions
			//vector too
			if (f->get_type() & instruction_function_type::TRAINABLE)
				model->train_funcs.push_back((trainable_function *)f);

			//free the placeholder vector
			free(data_buff);
		}

		//close the data stream as we are finished reading
		data_stream.close();

		//return the object at the model pointer
		return *model;
	}
	
	void network_model::calc_batch_gradient(float * d_x_batch, float * d_y_batch, size_t current_batch_size)
	{
		//set the first input layer to be the input
		float * d_in_layer = d_x_batch;

		//loop through each function in the model
		for (int i = 0; i < instructions.size(); i++) {
			//get a reference to the current instruction
			instruction_function * i_func = instructions[i];

			//get a reference to the output from this layer
			float * d_out_layer = i_func->get_out_vector();

			//reset the output vector to 0s for this layer
			cuda_safe_call(cudaMemset(d_out_layer, 0, sizeof(float) * i_func->output_shape.size() * current_batch_size));

			//run this layer with the given input
			i_func->run(d_in_layer, current_batch_size);

			//run the derivative function for this layer to calculate relevant
			//information for back propagation
			i_func->run_derivative(d_in_layer);

			//if the function is trainable
			if (i_func->get_type() & instruction_function_type::TRAINABLE) {
				//get a reference to the function as a trainiable function
				trainable_function * t_func = (trainable_function *)i_func;

				//if the function isn't locked, run the training derivative function
				//to calculate relevant information for training this function
				if (!t_func->locked())
					t_func->run_train_derivative(d_in_layer, current_batch_size);
			}

			//set the next input to the output from this layer
			d_in_layer = d_out_layer;
		}

		//calculate the loss between the expected and observed results
		cost_func->cost(instructions.back()->get_out_vector(), d_y_batch, current_batch_size);

		//the current derivative value at each step i.e df1/df2 * df2/df3 * ... * dfn-1/dfn
		//needs to be the size of the largest layer * batch size
		float * d_current_layer_cr_derivative;

		//allocate the current derivatives pointer
		allocate_device_float_pointer(&d_current_layer_cr_derivative, largest_layer_size * current_batch_size);

		//work out the derivative between the distributions
		cost_func->cost_derivative(instructions.back()->get_out_vector(), d_y_batch, current_batch_size);

		//copy the derivative from the cost function into the current derivatives vector
		copy_into_device_array(cost_func->get_derivative_vector(), d_current_layer_cr_derivative, cost_func->get_size() * current_batch_size, 0);

		//loop through each function in reverse order
		for (int i = instructions.size() - 1; i >= 0; i--) {
			//get a reference to the current instruction function
			instruction_function * i_func = instructions[i];

			//if the function is trainable...
			if (i_func->get_type() & instruction_function_type::TRAINABLE) {
				//get a trainable reference
				trainable_function * t_func = (trainable_function *)i_func;

				//if the function isn't locked, calculate the derivatives with reference to the 
				//training parameters
				if (!t_func->locked())
					t_func->avg_partial_derivatives(d_current_layer_cr_derivative, current_batch_size);
			}

			//if this isn't the last layer, find the derivatives for the previous layer for back
			//propagation
			if (i != 0)
				i_func->back_propagate(d_current_layer_cr_derivative, current_batch_size);
		}
		
		//dereference the placeholder pointer
		deallocate_device_float_pointer(d_current_layer_cr_derivative);
	}
}
