#include "stdafx.h"

#include "network_model.h"


namespace nn {
	network_model::network_model()
	{
	}


	network_model::~network_model()
	{
	}

	void network_model::entry(shape entry_shape)
	{
		this->init_layer_shape = entry_shape;
		__ent_spec = true;
	}

	void network_model::add(tensor biases)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		if (biases.get_dimensions() != 1)
			throw exception("Bias tensor must be one dimensional");

		add_function * f = new add_function(biases);
		instructions.push_back(f);
		train_funcs.push_back(f);
	}

	void network_model::mul(tensor weights)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		if (weights.get_dimensions() != 2)
			throw exception("Weight tensor must be two dimensional");
		mul_function * f = new mul_function(weights);
		instructions.push_back(f);
		train_funcs.push_back(f);
		this->init_layer_shape = shape(weights.get_shape()[0]);
	}

	/*void network_model::dense(size_t in_size, size_t out_size)
	{
		tensor weights = tensor::random({ out_size, in_size }, -0.1, 0.1);
		tensor biases = tensor::random(out_size, -0.1, 0.1);
		mul(weights);
		add(biases);
	}*/

	void network_model::dense(size_t units) {
		if (!__ent_spec)
			throw exception("Entry size not specified");

		tensor weights = tensor::random({ units, init_layer_shape.width }, -0.1, 0.1);
		tensor biases = tensor::random(units, -0.1, 0.1);
		mul(weights);
		add(biases);
		init_layer_shape = shape(units);
	}

	void network_model::conv2d(shape filter_shape, size_t n_filters, shape padding)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		tensor filter = tensor::random({ filter_shape.width, filter_shape.height, filter_shape.depth, n_filters }, -0.1, 0.1);
		conv2d(filter, padding);
	}

	void network_model::conv2d(tensor filter, shape padding)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		if (filter.get_dimensions() != 4)
			throw exception("Conv2d filter must be four dimensional (width, height, depth, filters)");
		conv2d_function * f = new conv2d_function(filter, padding);
		instructions.push_back(f);
		train_funcs.push_back(f);
		f->set_input_shape(init_layer_shape);
		init_layer_shape = f->output_shape;
	}

	void network_model::pool(shape pool_size, shape stride)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		pool_function * f = new pool_function(pool_size, stride);
		instructions.push_back(f);
		f->set_input_shape(init_layer_shape);
		init_layer_shape = f->output_shape;
	}

	void network_model::flatten()
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		flatten_function * f = new flatten_function(init_layer_shape);
		instructions.push_back(f);
		init_layer_shape = f->output_shape;
	}

	void network_model::reshape(shape output_shape)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		instructions.push_back(new reshape_function(init_layer_shape, output_shape));
		init_layer_shape = output_shape;
	}

	void network_model::relu()
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		instructions.push_back(new relu_function(init_layer_shape.size()));
	}

	void network_model::leaky_relu(float alpha)
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		instructions.push_back(new leaky_relu_function(init_layer_shape.size(), alpha));
	}

	void network_model::tanh()
	{
		if (!__ent_spec)
			throw exception("Entry size not specified");

		instructions.push_back(new tanh_function(init_layer_shape.size()));
	}

	/*void network_model::softmax()
	{
		instructions.push_back(new softmax_function());
	}

	void network_model::softmax(float beta)
	{
		instructions.push_back(new softmax_function(0, beta));
	}*/

	void network_model::function(instruction_function *func)
	{
		instructions.push_back(func);
	}
	
	void network_model::add_logger(analytics logger)
	{
		analytics_logger = &logger;
	}

	void network_model::initialise_model()
	{
		initialise_model(1);
	}

	/*void network_model::initialise_model(size_t batch_size) {
		initialise_model(instructions[0]->input_shape, batch_size);
	}*/

	void network_model::initialise_model(size_t batch_size)
	{
		if (instructions.size() == 0)
			throw std::exception("Cannot initialise empty model");

		//instructions[0]->set_input_shape(input_shape);
		//layer_shapes.push_back(instructions[0]->input_shape);

		//shape prev_layer_shape = instructions[0]->input_shape;

		largest_layer_size = -1; // prev_layer_shape.size();

		for (int i = 0; i < instructions.size(); i++) {
			/*if (instructions[i]->input_shape.size() == 0 ||
				instructions[i]->input_shape.size() == 1) {

				instructions[i]->set_input_shape(prev_layer_shape);
			}
			else if (instructions[i]->input_shape != prev_layer_shape)
				throw exception(("Layer tensor shape cannot change from " + prev_layer_shape.str() + " to " + instructions[i]->input_shape.str() + " without transformation").c_str());*/
			instructions[i]->initialise(batch_size);
			layer_shapes.push_back(instructions[i]->output_shape);
			//prev_layer_shape = instructions[i]->output_shape;

			//if (prev_layer_shape.size() > largest_layer_size)
			//	largest_layer_size = prev_layer_shape.size();
			if (instructions[i]->input_shape.size() > largest_layer_size)
				largest_layer_size = instructions[i]->input_shape.size();
		}

		output_shape = instructions.back()->output_shape;

		if (cost_func != nullptr)
			cost_func->initialise(output_shape, batch_size, -1);
		if (output_func != nullptr)
			output_func->initialise(output_shape, batch_size);

		this->batch_size = batch_size;

		model_initialised = true;
	}

	void network_model::uninitialise_model()
	{
		for (int i = 0; i < instructions.size(); i++) {
			instructions[i]->uninitialise();
		}

		if (cost_func != nullptr)
			cost_func->uninitialise();
		if (output_func != nullptr)
			output_func->uninitialise();

		model_initialised = false;
	}
	
	tensor network_model::run(tensor input)
	{
		if (!model_initialised)
			throw exception("Model not initialised");

		input.initialise();

		/*if (input.get_dimensions() != 1)
			throw exception("Input tensor must be one dimenional");

		if (input.get_shape()[0] != layer_sizes[0])
			throw exception(("Input tensor has size " + to_string(input.get_shape()[0]) + ", whereas model requires input of size " + to_string(layer_sizes[0])).c_str());*/
		
		float * d_in_batch, *d_in_layer, * d_out_layer, * d_output;
		d_out_layer = new float();

		//size_t last_size;

		if (input.get_dimensions() == 1)
			input.reshape({ 1, input.get_shape()[0] });

		size_t num = input.get_shape()[0];
		shape input_shape = layer_shapes[0];

		//cost_func->set_total_size(num);

		//allocate_device_pointer(&d_in_batch, batch_size * input_size);
		allocate_device_pointer(&d_output, num * output_shape.size());

		int n_batches = ceil(num / (float)batch_size);
		size_t current_batch_size = batch_size;

		for (int batch = 0; batch < n_batches; batch++) {

			if (batch == n_batches - 1) {
				current_batch_size = num % batch_size;
				if (current_batch_size == 0)
					current_batch_size = batch_size;
			}

			//load_data_into_device(&input.get_data()[batch * batch_size * input_size], d_in_batch, current_batch_size * input_size);

			d_in_batch = &input.get_dev_pointer()[batch * batch_size * input_shape.size()];

			d_in_layer = d_in_batch;

			for (int i = 0; i < instructions.size(); i++) {
				d_out_layer = instructions[i]->get_out_vector();
				//fill_array(d_out_layer, 0, batch_size * instructions[i]->output_shape.size());
				fill_device_array(d_out_layer, 0, instructions[i]->output_shape.size());
				instructions[i]->run(d_in_layer, current_batch_size);
				d_in_layer = d_out_layer;

				//last_size = instructions[i]->output_shape.size();
			}

			/*for (int i = 0; i < instructions.size(); i++) {
				float * test = (float *)malloc(sizeof(float) * 10);
				cudaMemcpy(test, instructions[i]->get_out_vector(), sizeof(float) * 10, cudaMemcpyDeviceToHost);
				for (int j = 0; j < 10; j++)
					printf("test[%d][%d] = %f\n", i, j, test[j]);
			}*/
			
			/*float * test = (float *)malloc(sizeof(float) * 20);
			cudaMemcpy(test, d_out_layer, sizeof(float) * 20, cudaMemcpyDeviceToHost);

			for (int k = 0; k < 20; k++)
				printf("Test[%d] = %f\n", k, test[k]);*/

			if (output_func != nullptr) {
				output_func->run(d_out_layer, current_batch_size);
				d_out_layer = output_func->get_out_vector();
			}

			copy_into_device_array(d_out_layer, d_output, current_batch_size * output_shape.size(), batch * batch_size * output_shape.size());
		}

		float * output = (float *)malloc(sizeof(float) * num * output_shape.size());
		retrieve_output_data(output, d_output, output_shape.size() * num);

		//deallocate_device_pointer(d_in_batch);
		deallocate_device_pointer(d_output);

		tensor * ret_tensor;

		if (output_shape.height == 1)
			ret_tensor = new tensor({ num, output_shape.width }, output);
		else
			ret_tensor = new tensor({ num, output_shape.width, output_shape.height }, output);

		return *ret_tensor;
	}
	
	void network_model::train(tensor train_x, tensor train_y, int epochs)
	{
		if (!model_initialised)
			throw exception("Model not initialised");

		train_x.initialise();
		train_y.initialise();

		//if (train_x.get_dimensions() == 1)
		//	train_x.reshape({ 1, train_x.get_shape()[0] });

		size_t num = train_x.get_shape()[0];
		shape input_shape = layer_shapes[0];

		cost_func->set_total_size(num);

		int n_batches = ceil(num / (float)batch_size);
		size_t current_batch_size = batch_size;

		if (analytics_logger != nullptr) {
			analytics_logger->init_logging();
		}

		for (int epoch = 0; epoch < epochs; epoch++) {
			if (analytics_logger != nullptr) {
				//analytics_logger->start_log();
				analytics_logger->on_epoch_start();
			}

			cost_func->clear_loss();
			
			for (int batch = 0; batch < n_batches; batch++) {
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_start();
				}

				if (batch == n_batches - 1) {
					current_batch_size = num % batch_size;
					if (current_batch_size == 0)
						current_batch_size = batch_size;
				}

				calc_batch_gradient(
					&train_x.get_dev_pointer()[batch * batch_size * input_shape.size()],
					&train_y.get_dev_pointer()[batch * batch_size * output_shape.size()],
					current_batch_size
				);

				for (int t_f = 0; t_f < train_funcs.size(); t_f++) {
					opt->optimise(train_funcs[t_f], epoch);
				}

				if (analytics_logger != nullptr) {
					analytics_logger->on_step_end();
				}
			}

			/*for (int i = 0; i < instructions.size(); i++) {
				if (instructions[i]->get_type() & instruction_function_type::TRAINABLE) {
					trainable_function * t_func = (trainable_function *)instructions[i];
					t_func->train_function(learning_rate);
				}
			}*/

			if (analytics_logger != nullptr) {
				//analytics_logger->stop_log();
				//analytics_logger->print_log(epoch + 1, cost_func->get_average_loss());
				analytics_logger->on_epoch_end();
			}
		}

		if (analytics_logger != nullptr) {
			analytics_logger->end_logging();
		}
	}

	void network_model::train(batch_iterator & b_iter, int epochs)
	{
		if (!model_initialised)
			throw exception("Model not initialised");

		b_iter.initialise(batch_size);

		size_t num = b_iter.get_size();
		shape input_size = layer_shapes[0];

		cost_func->set_total_size(num);

		int n_batches = ceil(num / (float)batch_size);
		size_t current_batch_size = batch_size;

		if (analytics_logger != nullptr) {
			analytics_logger->init_logging();
		}

		for (int epoch = 0; epoch < epochs; epoch++) {
			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_start();
			}

			cost_func->clear_loss();
			float epoch_loss = 0;

			for (int batch = 0; batch < n_batches; batch++) {
				if (analytics_logger != nullptr) {
					analytics_logger->on_step_start();
				}

				if (batch == n_batches - 1) {
					current_batch_size = num % batch_size;
					if (current_batch_size == 0)
						current_batch_size = batch_size;
				}

				b_iter.next_batch();
				tensor * train_x = b_iter.get_next_batch();
				tensor * train_y = b_iter.get_next_batch_labels();

				train_x->initialise();
				train_y->initialise();

				calc_batch_gradient(
					train_x->get_dev_pointer(),
					train_y->get_dev_pointer(),
					current_batch_size
				);

				int x = 3;

				for (int t_f = 0; t_f < train_funcs.size(); t_f++) {
					opt->optimise(train_funcs[t_f], epoch);
				}

				//printf("##########################################\n");
				
				epoch_loss += cost_func->get_average_loss() * current_batch_size;

				if (analytics_logger != nullptr) {
					analytics_logger->on_step_end(cost_func->get_average_loss());
				}
			}

			b_iter.reset_iterator();

			/*for (int i = 0; i < instructions.size(); i++) {
				if (instructions[i]->get_type() & instruction_function_type::TRAINABLE) {
					trainable_function * t_func = (trainable_function *)instructions[i];
					t_func->train_function(learning_rate);
				}
			}*/

			if (analytics_logger != nullptr) {
				analytics_logger->on_epoch_end(epoch_loss / num);
			}
		}

		if (analytics_logger != nullptr) {
			analytics_logger->end_logging();
		}
	}

	float network_model::get_accuracy(batch_iterator & b_iter)
	{
		if (!model_initialised)
			throw exception("Model not initialised");

		b_iter.initialise(batch_size);

		/*if (input.get_dimensions() != 1)
			throw exception("Input tensor must be one dimenional");

		if (input.get_shape()[0] != layer_sizes[0])
			throw exception(("Input tensor has size " + to_string(input.get_shape()[0]) + ", whereas model requires input of size " + to_string(layer_sizes[0])).c_str());*/

		float * d_in_batch, *d_in_layer, *d_out_layer;
		d_out_layer = new float();

		//size_t last_size;

		size_t num = b_iter.get_size();
		shape input_shape = layer_shapes[0];

		int total_correct = 0;

		//allocate_device_pointer(&d_in_batch, batch_size * input_size);

		int n_batches = ceil(num / (float)batch_size);
		size_t current_batch_size = batch_size;

		for (int batch = 0; batch < n_batches; batch++) {

			if (batch == n_batches - 1) {
				current_batch_size = num % batch_size;
				if (current_batch_size == 0)
					current_batch_size = batch_size;
			}

			b_iter.next_batch();
			d_in_batch = b_iter.get_next_batch()->get_dev_pointer();

			d_in_layer = d_in_batch;

			for (int i = 0; i < instructions.size(); i++) {
				d_out_layer = instructions[i]->get_out_vector();
				fill_device_array(d_out_layer, 0, batch_size * instructions[i]->output_shape.size());
				instructions[i]->run(d_in_layer, current_batch_size);
				d_in_layer = d_out_layer;
			}

			//calculate no. correct values
			
			//compare argmax to labels
			float * out_vals = (float *)malloc(sizeof(float) * current_batch_size * output_shape.size());
			cuda_safe_call(cudaMemcpy(out_vals, d_out_layer, sizeof(float) * current_batch_size * output_shape.size(), cudaMemcpyDeviceToHost));

			float * batch_labels = (float *)malloc(sizeof(float) * current_batch_size);
			cuda_safe_call(cudaMemcpy(batch_labels, b_iter.get_next_batch_labels()->get_dev_pointer(), sizeof(float) * current_batch_size, cudaMemcpyDeviceToHost));

			for (int res = 0; res < current_batch_size; res++) {
				float * start = &out_vals[res * output_shape.size()];
				float * end = start + output_shape.size();
				int res_argmax = distance(start, max_element(start, end));
				if (res_argmax == batch_labels[res])
					total_correct++;
			}
		}

		return total_correct / (float)num;
	}
	
	void network_model::write_model_to_file(string model_folder, string model_name)
	{
		CreateDirectoryA((LPCSTR)(model_folder + "\\" + model_name).c_str(), NULL);
		//_wmkdir((LPCWSTR)((model_folder + "\\" + model_name).c_str()));

		ofstream data_stream(model_folder + "\\" + model_name + "\\" + model_name + ".model", ofstream::binary);
		//ofstream meta_stream(model_folder + "\\" + model_name + "\\" + model_name + ".meta", ofstream::binary);

		int i_size = instructions.size();

		data_stream.write((char *)&i_size, sizeof(int));

		for (int layer = 0; layer < i_size; layer++) {
			instruction_function * i_func = instructions[layer];

			size_t stream_size = i_func->get_serialise_size();
			char * stream_buffer = (char *)malloc(stream_size);

			i_func->serialise(stream_buffer, 0);

			data_stream.write(reinterpret_cast<char*>(reinterpret_cast<void*>(&stream_size)), sizeof(size_t));
			data_stream.write(stream_buffer, stream_size);
		}

		//meta_stream.close();
		data_stream.close();
	}
	
	network_model network_model::load_model_from_file(string model_folder, string model_name)
	{
		network_model * model = new network_model();

		ifstream data_stream(model_folder + "\\" + model_name + "\\" + model_name + ".model", ifstream::binary);
		//ifstream meta_stream(model_folder + "\\" + model_name + "\\" + model_name + ".meta", ifstream::binary);

		data_stream.seekg(0, data_stream.end);
		size_t length = data_stream.tellg();
		data_stream.seekg(0, data_stream.beg);

		char n_layers_b[4];
		int n_layers;

		data_stream.read(n_layers_b, sizeof(int));
		n_layers = *(int *)n_layers_b;

		for (int layer = 0; layer < n_layers; layer++) {
			//meta_stream.read(layer_info_b, sizeof(int) * 3);
			size_t layer_stream_size;
			data_stream.read((char *)&layer_stream_size, sizeof(size_t));

			char * data_buff = (char *)malloc(layer_stream_size);
			data_stream.read(data_buff, layer_stream_size);

			function_id func_id = *reinterpret_cast<function_id*>(reinterpret_cast<void*>(&data_buff[0]));

			instruction_function * f;

			switch (func_id) {
			case function_id::ADD:
				f = new add_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::MUL:
				f = new mul_function();
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
			case function_id::BATCH_NORM:
				break;
			case function_id::CONV_2D:
				f = new conv2d_function();
				f->deserialise(data_buff, 0);
				break;
			case function_id::POOL:
				f = new pool_function();
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

			model->instructions.push_back(f);
			if (f->get_type() & instruction_function_type::TRAINABLE)
				model->train_funcs.push_back((trainable_function *)f);
		}

		//meta_stream.close();
		data_stream.close();

		return *model;
	}
	
	void network_model::calc_batch_gradient(float * d_x_batch, float * d_y_batch, size_t current_batch_size)
	{
		//size_t last_size;
		float * d_in_layer = d_x_batch;
		
		for (int i = 0; i < instructions.size(); i++) {

			instruction_function * i_func = instructions[i];
			float * d_out_layer = i_func->get_out_vector();

			fill_device_array(d_out_layer, 0, batch_size * i_func->output_shape.size());

			i_func->run(d_in_layer, current_batch_size);

			i_func->run_derivative(d_in_layer);

			if (i_func->get_type() & instruction_function_type::TRAINABLE) {
				trainable_function * t_func = (trainable_function *)i_func;
				t_func->run_train_derivative(d_in_layer, current_batch_size);

				/*float * test = (float *)malloc(sizeof(float) * 10);
				cudaMemcpy(test, d_out_layer, sizeof(float) * 10, cudaMemcpyDeviceToHost);

				for (int i = 0; i < 10; i++)
					printf("test[%d] = %.32g\n", i, test[i]);
				printf("\n");

				double total = 0;

				float * A = (float *)malloc(sizeof(float) * 784);
				cudaMemcpy(A, t_func->get_train_vector(), sizeof(float) * 784, cudaMemcpyDeviceToHost);

				float * B = (float *)malloc(sizeof(float) * 784);
				cudaMemcpy(B, d_in_layer, sizeof(float) * 784, cudaMemcpyDeviceToHost);

				for (int i = 0; i < 784; i++)
					total += A[i] * B[i];
				printf("total: %.32g\n", total);
				printf("\n");*/

			}

			d_in_layer = d_out_layer;

			//last_size = instructions[i]->output_size;
		}
		
		/*float * test = (float *)malloc(sizeof(float) * 30);
		cudaMemcpy(test, instructions.back()->get_out_vector(), sizeof(float) * 30, cudaMemcpyDeviceToHost);

		for (int k = 0; k < 30; k++)
			printf("test[%d] = %e\n", k, test[k]);
		printf("\n");/**/

		cost_func->cost(instructions.back()->get_out_vector(), d_y_batch, current_batch_size);
		
		//after forward pass, each of the layers still has reference to the output from that layer.

		//the current derivative value at each step i.e df1/df2 * df2/df3 * ... * dfn-1/dfn
		//needs to be the size of the largest layer * batch size
		float * d_current_layer_cr_derivative;
		//float * d_temp;

		allocate_device_pointer(&d_current_layer_cr_derivative, largest_layer_size * current_batch_size);
		//allocate_device_pointer(&d_temp, largest_layer_size * current_batch_size);

		cost_func->cost_derivative(instructions.back()->get_out_vector(), d_y_batch, current_batch_size);

		copy_into_device_array(cost_func->get_derivative_vector(), d_current_layer_cr_derivative, cost_func->get_size() * current_batch_size, 0);
		
		for (int i = instructions.size() - 1; i >= 0; i--) {

			/*float * test = (float *)malloc(sizeof(float) * 10);
			cudaMemcpy(test, d_current_layer_cr_derivative, sizeof(float) * 10, cudaMemcpyDeviceToHost);

			for (int i = 0; i < 10; i++)
				printf("test[%d] = %e\n", i, test[i]);
			printf("\n");*/

			instruction_function * i_func = instructions[i];

			if (i_func->get_type() & instruction_function_type::TRAINABLE) {
				trainable_function * t_func = (trainable_function *)i_func;
				t_func->avg_partial_derivatives(d_current_layer_cr_derivative, current_batch_size);
			}

			if (i != 0)
				i_func->back_propagate(d_current_layer_cr_derivative, current_batch_size);

			//needs to move
			/*if (i_func->get_type() & instruction_function_type::INTER_DIM_TRANSFORM) {
				matrix_multiply<float, order::COL, order::COL, order::COL>(
					d_matrix<float>({ (int)i_func->output_shape.width, (int)i_func->input_shape.width, i_func->get_derivative_vector() }),
					d_matrix<float>({ (int)current_batch_size, (int)i_func->output_shape.width, d_current_layer_cr_derivative }),
					d_matrix<float>({ (int)current_batch_size, (int)i_func->input_shape.width, d_temp }));

				//cuda_safe_call(cudaMemcpy(d_current_layer_cr_derivative, d_temp, sizeof(float) * current_batch_size * i_func->input_size, cudaMemcpyDeviceToDevice));
				copy_into_device_array(d_temp, d_current_layer_cr_derivative, current_batch_size * i_func->input_shape.width, 0);
			}
			else {
				hadamard_product(d_current_layer_cr_derivative, i_func->get_derivative_vector(), d_current_layer_cr_derivative,
					i_func->input_shape.size() * current_batch_size);
			}*/
		}
		
		deallocate_device_pointer(d_current_layer_cr_derivative);
		//deallocate_device_pointer(d_temp);
	}
}
