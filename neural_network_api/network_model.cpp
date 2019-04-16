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
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create an add function
		add_function* f = new add_function(biases);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		return f;
	}

	matmul_function* network_model::matmul(tensor weights)
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//check the tensor is a matrix
		ERR_ASSERT(weights.get_dimensions() != 2, "Weight tensor must be two dimensional");

		//create a matmul function
		matmul_function* f = new matmul_function(weights);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		//set the shape of the next layer to be the output shape from the matrix
		this->init_layer_shape = shape(weights.get_shape()[0]);

		return f;
	}

	dense_layer network_model::dense(size_t units, variable_initialiser weight_initialiser, variable_initialiser bias_initialiser) {
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

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
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

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
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

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
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//check the filter has the right number of dimensions
		ERR_ASSERT(filter.get_dimensions() != 4, "Conv2d filter must be four dimensional (width, height, depth, filters)");

		//create the conv2d function
		conv2d_function * f = new conv2d_function(filter, padding);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

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
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the max pool function
		max_pool_function * f = new max_pool_function(pool_size, stride);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

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
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the flatten function 
		flatten_function * f = new flatten_function(init_layer_shape);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		//set the next layer shape to be the output shape
		init_layer_shape = f->output_shape;

		return f;
	}

	reshape_function* network_model::reshape(shape output_shape)
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the reshape function
		reshape_function * f = new reshape_function(init_layer_shape, output_shape);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		//set the next layer shape to be the output shape
		init_layer_shape = output_shape;

		return f;
	}

	dropout_function* network_model::dropout(float keep_rate)
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the reshape function
		dropout_function* f = new dropout_function(keep_rate);

		f->set_input_shape(init_layer_shape);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		return f;
	}

	relu_function* network_model::relu()
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the relu function 
		relu_function* f = new relu_function(init_layer_shape);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		return f;
	}

	leaky_relu_function* network_model::leaky_relu(float alpha)
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");
		
		//create the leaky relu function 
		leaky_relu_function* f = new leaky_relu_function(init_layer_shape, alpha);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		return f;
	}

	tanh_function* network_model::tanh()
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the tanh function 
		tanh_function* f = new tanh_function(init_layer_shape);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		return f;
	}

	sigmoid_function* network_model::sigmoid()
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//create the sigmoid function 
		sigmoid_function* f = new sigmoid_function(init_layer_shape);

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(f));

		return f;
	}

	instruction_function* network_model::function(instruction_function *func)
	{
		//check that the entry shape is specified
		ERR_ASSERT(!__ent_spec, "Entry size not specified");

		//push the function to the initialiser list
		graph_nodes.push_back(node<instruction_function*>(func));

		return func;
	}
	
	void network_model::add_logger(analytics& logger)
	{
		analytics_logger = &logger;
	}
	
	void network_model::initialise_model(size_t batch_size)
	{
		//if the model is already initialised abort and return
		if (model_initialised)
			return;

		//check that there is a model to initialise
		ERR_ASSERT(graph_nodes.size() == 0, "Cannot initialise empty model");

		model_graph.add_node(&graph_nodes[0]);

		//set the graph entry
		model_graph.set_start_point(&graph_nodes[0]);

		//set up edges
		for (int i = 1; i < graph_nodes.size(); i++) {
			//get a reference to the current and previous nodes
			node<instruction_function*>* n = &graph_nodes[i];
			node<instruction_function*>* n_p = &graph_nodes[i - 1];

			//provide references for the nodes children and parents
			n->parents.push_back(n_p);
			n_p->children.push_back(n);

			//add this node to the graph
			model_graph.add_node(n);
		}

		//get a reference to the output shape from the model
		output_shape = graph_nodes.back().value->output_shape;

		//set the graph exit
		model_graph.set_end_point(&graph_nodes.back());

		//initialise the graph
		model_graph.initialise(output_shape, batch_size);

		//if there is an optimiser, initialise it
		if (opt != nullptr)
			opt->initialise(&model_graph);

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

		//if there was a optimiser, uninitialise it
		if (opt != nullptr)
			opt->uninitialise();

		//flag that the model is now uninitialised
		model_initialised = false;
	}
	
	tensor network_model::run(tensor input)
	{
		//if the model is not initialised it can not be run, so throw an exception
		ERR_ASSERT(!model_initialised, "Model not initialised");

		//initialise the input tensor in case it has not been initialised
		input.initialise();

		//declare pointers for batches, layers and outputs
		float * d_in_batch, * d_output;

		//get the number of inputs, which should be in the first dimension of the input tensor
		size_t num = input.get_shape()[0];

		//allocate a placeholder pointer for the output of each batch
		allocate_device_float_pointer(&d_output, num * model_graph.get_output_shape().size());

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
			d_in_batch = &input.get_dev_pointer()[batch * batch_size * model_graph.get_input_shape().size()];

			//run the graph with the input data
			run_graph(
				model_graph, 
				d_in_batch, 
				&d_output[batch * batch_size * model_graph.get_output_shape().size()], 
				current_batch_size
			);
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
	
	void network_model::train(tensor train_x, tensor train_y, int epochs, bool from_start)
	{
		//if the model is not initialised it can not be run, so throw an exception
		ERR_ASSERT(!model_initialised, "Model not initialised");

		//if this model should train as if it was just started, set the step counter to 0
		if (from_start)
			__step = 0;

		//initialise the input tensors in case they have not been initialised
		train_x.initialise();
		train_y.initialise();

		//get the number of inputs, which should be in the first dimension of the input tensor
		size_t num = train_x.get_shape()[0];

		//calculate the number of batches depending on the number of inputs and the batch size
		int n_batches = ceil(num / (float)batch_size);

		//begin with the current batch size being the primary batch size
		size_t current_batch_size = batch_size;

		//if there is a logger on the model, begin logging
		if (analytics_logger != nullptr) {
			analytics_logger->init_logging();
		}

		//get the starting epoch if we are not starting from scratch
		int start_epoch = __step / n_batches;

		//get the starting batch step if we are not starting from scratch
		int start_batch = __step % n_batches;

		//loop through each epoch. Every epoch represents training against the entire tensors
		//specified
		for (int epoch = start_epoch; epoch < epochs; epoch++) {
			//if there is a logger, call its "on epoch start" event, as this epoch is
			//starting
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_start();
			}

			//clear the loss metric in the cost function
			model_graph.get_cost_function()->clear_loss();

			//set the total loss for this epoch to 0
			float epoch_loss = 0;

			//set the current batch size to the primary batch size
			current_batch_size = batch_size;

			//loop through each batch
			for (int batch = start_batch; batch < n_batches; batch++) {
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
				calculate_gradients(
					model_graph,
					&train_x.get_dev_pointer()[batch * batch_size * model_graph.get_input_shape().size()],
					&train_y.get_dev_pointer()[batch * batch_size * output_shape.size()],
					current_batch_size,
					METRIC_LOSS
				);

				//optimise the trainable functions in the network with the optimiser
				opt->optimise();

				//add the total loss for this batch to the epoch loss
				epoch_loss += model_graph.get_cost_function()->get_average_loss() * current_batch_size;

				//if there is a logger, call the "on step end" event function, as this step
				//has just ended
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_end(model_graph.get_cost_function()->get_average_loss());
				}

				__step++;

				//if this is a checkpoint step
				if (cpt & checkpoint_type::CHECKPOINT_PER_STEPS) {
					if (__step % cpt_steps == 0)
						__export_model_to_file(file_path, model_name);
				}
			}

			//reset the starting batch so we don't miss necessary batches
			start_batch = 0;

			//if there is a logger, call the "on epoch end" event function, as this epoch has just
			//ended
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_end(epoch_loss / num);
			}

			//if there is epoch checkpointing
			if (cpt & checkpoint_type::CHECKPOINT_PER_EPOCH) {
				__export_model_to_file(file_path, model_name);
			}
		}

		//if there is a logger, call the end_logging() function to stop logging
		if (analytics_logger != nullptr) {
			analytics_logger->end_logging();
		}

		//if saving at the end of the model is chosen
		if (cpt & checkpoint_type::CHECKPOINT_END) {
			__export_model_to_file(file_path, model_name);
		}
	}

	void network_model::train(batch_iterator & b_iter, int epochs, bool from_start)
	{
		//if the model is not initialised it can not be run, so throw an exception
		ERR_ASSERT(!model_initialised, "Model not initialised");

		//if this model should train as if it was just started, set the step counter to 0
		if (from_start)
			__step = 0;

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

		//get the starting epoch if we are not starting from scratch
		int start_epoch = __step / n_batches;

		//get the starting batch step if we are not starting from scratch
		int start_batch = __step % n_batches;

		//loop through each epoch. Every epoch represents training against the entire iterator
		//specified
		for (int epoch = start_epoch; epoch < epochs; epoch++) {
			//if there is a logger, call its "on epoch start" event, as this epoch is
			//starting
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_start();
			}

			//clear the loss metric in the cost function
			model_graph.get_cost_function()->clear_loss();

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
				calculate_gradients(
					model_graph,
					train_x->get_dev_pointer(),
					train_y->get_dev_pointer(),
					current_batch_size,
					METRIC_LOSS
				);

				//optimise the trainable functions in the network with the optimiser
				opt->optimise();

				//add the total loss for this batch to the epoch loss
				epoch_loss += model_graph.get_cost_function()->get_average_loss() * current_batch_size;

				//if there is a logger, call the "on step end" event function, as this step
				//has just ended
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_end(model_graph.get_cost_function()->get_average_loss());
				}

				__step++;

				//if this is a checkpoint step
				if (cpt & checkpoint_type::CHECKPOINT_PER_STEPS) {
					if (__step % cpt_steps == 0)
						__export_model_to_file(file_path, model_name);
				}
			}

			//reset the starting batch so we don't miss necessary batches
			start_batch = 0;

			//reset the iterator back to the beginning ready for the next epoch
			b_iter.reset_iterator();

			//if there is a logger, call the "on epoch end" event function, as this epoch has just
			//ended
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_end(epoch_loss / num);
			}

			//if there is epoch checkpointing
			if (cpt & checkpoint_type::CHECKPOINT_PER_EPOCH) {
				__export_model_to_file(file_path, model_name);
			}
		}

		//if there is a logger, call the end_logging() function to stop logging
		if (analytics_logger != nullptr) {
			analytics_logger->end_logging();
		}

		//if saving at the end of the model is chosen
		if (cpt & checkpoint_type::CHECKPOINT_END) {
			__export_model_to_file(file_path, model_name);
		}
	}

	float network_model::evaluate(tensor test_x, tensor test_y)	{
		//if the model is not initialised it can not be run, so throw an exception
		ERR_ASSERT(!model_initialised, "Model not initialised");

		//initialise the input tensors in case they have not been initialised
		test_x.initialise();
		test_y.initialise();

		//declare pointers for batches, layers and outputs
		float * d_in_batch, *d_out_layer;

		//initialise the output layer placeholder
		allocate_device_float_pointer(&d_out_layer, batch_size * output_shape.size());

		//get the number of inputs, which should be in the first dimension of the input tensor
		size_t num = test_x.get_shape()[0];

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
			d_in_batch = &test_x.get_dev_pointer()[batch * batch_size * model_graph.get_input_shape().size()];

			run_graph(
				model_graph,
				d_in_batch,
				d_out_layer,
				current_batch_size
			);

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
		ERR_ASSERT(!model_initialised, "Model not initialised");

		//initialise the iterator with the same batch size as the model
		b_iter.initialise(batch_size);

		//declare pointers for batches, layers and outputs
		float * d_in_batch, *d_out_layer;

		//initialise the output layer placeholder
		allocate_device_float_pointer(&d_out_layer, batch_size * output_shape.size());

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

			run_graph(
				model_graph,
				d_in_batch,
				d_out_layer,
				current_batch_size
			);

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

	void network_model::export_model_to_file(string model_folder, string model_name)
	{
		//call the internal export function
		__export_model_to_file(model_folder, model_name);
	}
	
	network_model network_model::import_model_from_file(string model_folder, string model_name)
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
			instruction_function * f = nullptr;

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
			model->graph_nodes.push_back(node<instruction_function*>(f));

			//free the placeholder vector
			free(data_buff);
		}

		//close the data stream as we are finished reading
		data_stream.close();
		
		//open the checkpoint file
		ifstream checkpoint_stream(model_folder + "\\" + model_name + "\\" + model_name + ".checkpoint", ifstream::binary);

		//check that the file exists to compensate for older versions without checkpointing
		if (!checkpoint_stream.fail()) {
			//get the total length of the file
			checkpoint_stream.seekg(0, checkpoint_stream.end);
			size_t cpt_length = checkpoint_stream.tellg();
			checkpoint_stream.seekg(0, checkpoint_stream.beg);

			//if the file contains at least an int, read it into steps
			if (cpt_length > sizeof(int)) {
				checkpoint_stream.read((char*)model->__step, sizeof(int));
			}
		}

		//return the object at the model pointer
		return *model;
	}
	
	void network_model::set_checkpoint(int c_type, string model_folder, string model_name, int steps)
	{
		//set parameters from input
		this->cpt = c_type;
		this->file_path = model_folder;
		this->model_name = model_name;
		this->cpt_steps = steps;
	}

	void network_model::__export_model_to_file(string model_folder, string model_name)
	{
		//create the directory to save the model in (only works for Windows afaik)
		CreateDirectoryA((LPCSTR)(model_folder + "\\" + model_name).c_str(), NULL);

		//create a byte stream to write out to the .model file
		ofstream data_stream(model_folder + "\\" + model_name + "\\" + model_name + ".model", ofstream::binary);

		//get the number of instructions in the model
		int i_size = graph_nodes.size();

		//write the number of instructions to the beginning of the file
		data_stream.write((char*)& i_size, sizeof(int));

		//loop through each layer in the model
		for (int layer = 0; layer < i_size; layer++) {
			//get a temporary reference to the layer function
			instruction_function* i_func = graph_nodes[layer].value;

			//get the size which this function will stream to (in bytes)
			size_t stream_size = i_func->get_serialise_size();

			//allocate a buffer to serialise the function into
			char* stream_buffer = (char*)malloc(stream_size);

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

		//create a byte stream to write out to the .checkpoint file
		ofstream checkpoint_stream(model_folder + "\\" + model_name + "\\" + model_name + ".checkpoint", ofstream::binary);

		//write the current step
		checkpoint_stream.write((char*)&__step, sizeof(int));

		//close the filestream
		checkpoint_stream.close();
	}
}
