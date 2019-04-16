#include "stdafx.h"

#include "mnist_data_loader.h"

namespace nnet {
	mnist_data_loader::mnist_data_loader(string file_path, bool one_hot, int classes)
	{
		//setup variables from input
		this->file_path = file_path;
		this->one_hot = one_hot;
		this->n_classes = classes;
	}

	mnist_data_loader::~mnist_data_loader()
	{
	}

	void mnist_data_loader::load_data_set(string file_name)
	{
		//buffer for the metadata
		char buffer[16];

		//open the file stream
		ifstream data_stream = ifstream(file_path + "\\" + file_name, ios::binary);
		ERR_ASSERT(data_stream.fail(), "Failed to load file " << file_name);

		//if the file was successfully loaded, save the string
		d_file_name = file_name;

		//get the length of the file
		data_stream.seekg(0, data_stream.end);
		size_t length = data_stream.tellg();
		data_stream.seekg(0, data_stream.beg);

		//the file should be longer than the metadata size, otherwise it isn't in the MNIST
		//format. If it is longer, set the load length to the buffer size (as we only care about
		//the metadata at the moment)
		ERR_ASSERT(length <= sizeof(buffer), "Incorrect dataset file format");
		length = sizeof(buffer);

		//read in the metadata
		data_stream.read(buffer, length);

		//get the magic number from the first 4 bytes
		magic_num = read_int(&buffer[0]);

		//check that the dataset has the same number of items in the images and labels
		//otherwise throw an excpetion
		ERR_ASSERT(n_items != -1 && read_int(&buffer[4]) != n_items, "Data set has different number of data and labels");
		n_items = read_int(&buffer[4]);

		//read the number of rows from the 3rd 4 bytes
		n_rows = read_int(&buffer[8]);

		//read the number of cols from the 4th 4 bytes
		n_cols = read_int(&buffer[12]);

		//close the stream
		data_stream.close();

		//set the file index to the buffer length, so when we read next we don't
		//read the metadata again
		d_s_index = length;
	}

	void mnist_data_loader::load_data_set_labels(string file_name)
	{
		//buffer for the metadata
		char buffer[8];

		//open the file stream
		ifstream label_stream = ifstream(file_path + "\\" + file_name, ios::binary);
		ERR_ASSERT(label_stream.fail(), "Failed to load file " << file_name);

		//if the file was successfully loaded, save the string
		l_file_name = file_name;

		//get the length of the file
		label_stream.seekg(0, label_stream.end);
		size_t length = label_stream.tellg();
		label_stream.seekg(0, label_stream.beg);

		//the file should be longer than the metadata size, otherwise it isn't in the MNIST
		//format. If it is longer, set the load length to the buffer size (as we only care about
		//the metadata at the moment)
		ERR_ASSERT(length <= sizeof(buffer), "Incorrect dataset file format");
		length = sizeof(buffer);

		//read in the metadata
		label_stream.read(buffer, length);

		//get the magic number from the first 4 bytes
		magic_num_labels = read_int(&buffer[0]);

		//check that the dataset has the same number of items in the images and labels
		//otherwise throw an excpetion
		ERR_ASSERT(n_items != -1 && read_int(&buffer[4]) != n_items, "Data set has different number of data and labels");
		n_items = read_int(&buffer[4]);

		//close the stream
		label_stream.close();

		//set the file index to the buffer length, so when we read next we don't
		//read the metadata again
		l_s_index = length;
	}

	void mnist_data_loader::close()
	{
		//if not intialised, can't deallocate pointers so just return
		if (!initialised)
			return;

		//deallocate both device and host pointers
		cuda_safe_call(cudaFree(d_data_buffer));
		free(data_buffer);
		free(label_buffer);
		free(onehot_labels);
		free(index_labels);

		//uninitialise tensors
		data->uninitialise();
		labels->uninitialise();

		//flag that the loader is no longer initialised
		initialised = false;
	}

	tensor* mnist_data_loader::get_next_batch()
	{
		//open the data stream
		ifstream data_stream = ifstream(file_path + "\\" + d_file_name, ios::binary);

		//get the remaining length
		data_stream.seekg(0, data_stream.end);
		size_t length = (size_t)data_stream.tellg() - d_s_index;
		data_stream.seekg(d_s_index);

		//if the length is more than a full batch, set the tensor size to be the full batch
		//otherwise set the tensor to be the remainder batch size and fill the data buffer with 0s
		if (length >= sizeof(char) * batch_size * n_rows * n_cols) {
			length = sizeof(char) * batch_size * n_rows * n_cols;
			data->set_shape({ batch_size, n_rows * n_cols });
		}
		else {
			memset(data_buffer, 0, sizeof(char) * batch_size * n_rows * n_cols);
			data->set_shape({ n_items % batch_size, n_rows * n_cols });
		}

		//read in the data from the stream into the buffer
		data_stream.read(data_buffer, length);

		//copy the streamed data to the device
		cuda_safe_call(cudaMemcpy(d_data_buffer, data_buffer, sizeof(char) * batch_size * n_rows * n_cols, cudaMemcpyHostToDevice));

		//scale the data so it is normalised between 0 and 1, rather than 0 and 255
		scalar_matrix_multiply_b(d_data_buffer, data->get_dev_pointer(), 1.0 / 255.0, batch_size * n_rows * n_cols);

		//close the stream
		data_stream.close();

		//update the file index so we don't read the same information
		d_s_index += length;

		//return the data tensor from this batch
		return data;
	}

	tensor* mnist_data_loader::get_next_batch_labels()
	{
		//open the data stream
		ifstream label_stream = ifstream(file_path + "\\" + l_file_name, ios::binary);

		//get the remaining length
		label_stream.seekg(0, label_stream.end);
		size_t length = (size_t)label_stream.tellg() - l_s_index;
		label_stream.seekg(l_s_index);

		//initialise the current batch size to the primary batch size
		size_t current_batch_size = batch_size;

		//if we want one hot encoding...
		if (one_hot) {
			//check if this batch is all in the file
			if (length >= sizeof(char) * batch_size) {
				length = sizeof(char) * batch_size;
				labels->set_shape({ batch_size, n_classes });
			}
			else {
				//if not, fill the label buffer with 0s and decrease the current batch size
				current_batch_size = n_items % batch_size;

				memset(label_buffer, 0, sizeof(char) * batch_size);
				data->set_shape({ current_batch_size, n_classes });
			}

			//fill the one-hot labels with 0s
			memset(onehot_labels, 0, sizeof(float) * batch_size * n_classes);

			//read the batch from the stream into the buffer
			label_stream.read(label_buffer, length);

			//convert each element into one-hot encoding
			for (int i = 0; i < current_batch_size; i++) {
				onehot_labels[n_classes * i + label_buffer[i]] = 1;
			}

			//copy the one-hot labels into the labels tensor
			cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), onehot_labels, sizeof(float) * batch_size * n_classes, cudaMemcpyHostToDevice));

			//close the stream
			label_stream.close();

			//update the file index so we don't read the same information
			l_s_index += length;
		}
		else {
			//check if this batch is all in the file
			if (length >= sizeof(char) * batch_size) {
				length = sizeof(char) * batch_size;
				labels->set_shape({ (size_t)batch_size });
			}
			else {
				//if not, fill the label buffer with 0s and decrease the current batch size
				current_batch_size = n_items % batch_size;

				memset(label_buffer, 0, sizeof(char) * batch_size);
				data->set_shape({ (size_t)current_batch_size });
			}

			//fill the sparse labels with 0s
			memset(index_labels, 0, sizeof(float) * batch_size);

			//read the batch from the stream into the buffer
			label_stream.read(label_buffer, length);

			//write each element as a float into the index buffer
			for (int i = 0; i < current_batch_size; i++) {
				index_labels[i] = label_buffer[i];
			}

			//copy the index buffer to the label tensor
			cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), index_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice));

			//close the stream
			label_stream.close();

			//update the file index so we don't read the same information
			l_s_index += length;
		}

		//return the label tensor from this batch
		return labels;
	}

	void mnist_data_loader::reset_iterator()
	{
		//reset the file indices to the start, but after the metadata
		//(as we don't need to read this after each epoch)
		d_s_index = 16;
		l_s_index = 8;
	}

	void mnist_data_loader::initialise(size_t batch_size)
	{
		//if we are already initialised, abort and return
		if (initialised)
			return;

		//set the batch size
		this->batch_size = batch_size;

		//initialise the data buffers on host and device memory
		data_buffer = (char*)malloc(sizeof(char) * batch_size * n_rows * n_cols);
		cuda_safe_call(cudaMallocManaged(&d_data_buffer, sizeof(char) * batch_size * n_rows * n_cols));

		//create the data tensor of the correct size and initialise
		data = new tensor({ batch_size, n_rows * n_cols });
		data->initialise();

		//initialise label buffers on host memory
		label_buffer = (char*)malloc(sizeof(char) * batch_size);
		onehot_labels = (float*)malloc(sizeof(float) * batch_size * n_classes);
		index_labels = (float*)malloc(sizeof(float) * batch_size);

		//create and initialise the label tensor with the correct size and initialise, 
		//depending on whether it is one-hot encoded or not
		if (one_hot) {
			labels = new tensor({ batch_size, n_classes });
			labels->initialise();
			cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size * n_classes));
		}
		else {
			labels = new tensor({ batch_size });
			labels->initialise();
			cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size));
		}

		//flag that this loader is initialised
		initialised = true;
	}

	int mnist_data_loader::read_int(char* buff)
	{
		//reverse the byte stream order
		char b_r[4] = { buff[3], buff[2], buff[1], buff[0] };

		//static cast to an integer
		return *static_cast<int*>(static_cast<void*>(b_r));
	}
}