#include "stdafx.h"

#include "mnist_data_loader.h"

mnist_data_loader::mnist_data_loader(string file_path, bool one_hot)
	: mnist_data_loader(file_path, 0, true)
{
}

mnist_data_loader::mnist_data_loader(string file_path, size_t batch_size, bool one_hot)
{
	this->file_path = file_path;
	this->batch_size = batch_size;
	this->one_hot = one_hot;
}

mnist_data_loader::~mnist_data_loader()
{
}

void mnist_data_loader::load_data_set(string file_name)
{
	char buffer[16];

	ifstream data_stream = ifstream(file_path + "\\" + file_name, ios::binary);
	d_file_name = file_name;

	data_stream.seekg(0, data_stream.end);
	size_t length = data_stream.tellg();
	data_stream.seekg(0, data_stream.beg);

	if (length > sizeof(buffer)) {
		length = sizeof(buffer);
	}
	else {
		throw new exception("Incorrect dataset file format");
	}

	data_stream.read(buffer, length);

	data_stream.seekg(length);

	magic_num = read_int(&buffer[0]);
	if (n_items == -1)
		n_items = read_int(&buffer[4]);
	else if (read_int(&buffer[4]) != n_items)
		throw new exception("Data set has different number of data and labels.");
	n_rows = read_int(&buffer[8]);
	n_cols = read_int(&buffer[12]);

	data_stream.close();
	d_s_index += length;
}

void mnist_data_loader::load_data_set_labels(string file_name)
{
	char buffer[8];

	ifstream label_stream = ifstream(file_path + "\\" + file_name, ios::binary);
	l_file_name = file_name;

	label_stream.seekg(0, label_stream.end);
	size_t length = label_stream.tellg();
	label_stream.seekg(0, label_stream.beg);

	if (length > sizeof(buffer)) {
		length = sizeof(buffer);
	}
	else {
		throw new exception("Incorrect label file format");
	}

	label_stream.read(buffer, length);

	magic_num_labels = read_int(&buffer[0]);
	if (n_items == -1)
		n_items = read_int(&buffer[4]);
	else if (read_int(&buffer[4]) != n_items)
		throw new exception("Data set has different number of data and labels.");

	label_stream.close();
	l_s_index += length;
}

void mnist_data_loader::close()
{
	cuda_safe_call(cudaFree(d_data_buffer));
	free(data_buffer);
	free(label_buffer);
	free(onehot_labels);
	free(index_labels);
	initialised = false;
}

tensor * mnist_data_loader::get_next_batch()
{
	ifstream data_stream = ifstream(file_path + "\\" + d_file_name, ios::binary);

	data_stream.seekg(0, data_stream.end);
	size_t length = (size_t)data_stream.tellg() - d_s_index;
	data_stream.seekg(d_s_index);

	if (length >= sizeof(char) * batch_size * n_rows * n_cols) {
		length = sizeof(char) * batch_size * n_rows * n_cols;
		data->set_shape({ (size_t)batch_size, (size_t)(n_rows * n_cols) });
	}
	else {
		memset(data_buffer, 0, sizeof(char) * batch_size * n_rows * n_cols);
		data->set_shape({ (size_t)(n_items % batch_size), (size_t)(n_rows * n_cols) });		
	}
	
	data_stream.read(data_buffer, length);

	cuda_safe_call(cudaMemcpy(d_data_buffer, data_buffer, sizeof(char) * batch_size * n_rows * n_cols, cudaMemcpyHostToDevice));
	scalar_matrix_multiply<unsigned char, float>(d_data_buffer, data->get_dev_pointer(), 1.0 / 255.0, batch_size * n_rows * n_cols);

	data_stream.close();
	d_s_index += length;

	return data;
}

tensor * mnist_data_loader::get_next_batch_labels()
{
	ifstream label_stream = ifstream(file_path + "\\" + l_file_name, ios::binary);

	label_stream.seekg(0, label_stream.end);
	size_t length = (size_t)label_stream.tellg() - l_s_index;
	label_stream.seekg(l_s_index);

	int current_batch_size = batch_size;

	if (one_hot) {
		if (length >= sizeof(char) * batch_size) {
			length = sizeof(char) * batch_size;
			labels->set_shape({ (size_t)batch_size, 10 });
		}
		else {
			current_batch_size = n_items % batch_size;

			memset(label_buffer, 0, sizeof(char) * batch_size);
			data->set_shape({ (size_t)current_batch_size, 10 });
		}

		memset(onehot_labels, 0, sizeof(float) * batch_size * 10);

		label_stream.read(label_buffer, length);

		for (int i = 0; i < current_batch_size; i++) {
			onehot_labels[10 * i + label_buffer[i]] = 1;
		}

		cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), onehot_labels, sizeof(float) * batch_size * 10, cudaMemcpyHostToDevice));

		label_stream.close();
		l_s_index += length;

		return labels;
	}
	else {
		if (length >= sizeof(char) * batch_size) {
			length = sizeof(char) * batch_size;
			labels->set_shape({ (size_t)batch_size });
		}
		else {
			current_batch_size = n_items % batch_size;

			memset(label_buffer, 0, sizeof(char) * batch_size);
			data->set_shape({ (size_t)current_batch_size });
		}

		memset(index_labels, 0, sizeof(float) * batch_size);

		label_stream.read(label_buffer, length);

		for (int i = 0; i < current_batch_size; i++) {
			index_labels[i] = label_buffer[i];
		}

		cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), index_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice));

		label_stream.close();
		l_s_index += length;

		return labels;
	}
}

void mnist_data_loader::reset_iterator()
{
	d_s_index = 16;
	l_s_index = 8;
}

void mnist_data_loader::initialise(size_t batch_size)
{
	if (initialised)
		return;

	this->batch_size = batch_size;

	//initialise data reader
	data_buffer = (char *)malloc(sizeof(char) * batch_size * n_rows * n_cols);
	cuda_safe_call(cudaMallocManaged(&d_data_buffer, sizeof(char) * batch_size * n_rows * n_cols));

	data = new tensor({ (size_t)(batch_size), (size_t)(n_rows * n_cols) });
	data->initialise();

	//initialise label reader
	label_buffer = (char *)malloc(sizeof(char) * batch_size);
	onehot_labels = (float *)malloc(sizeof(float) * batch_size * 10);
	index_labels = (float *)malloc(sizeof(float) * batch_size);

	if (one_hot) {
		labels = new tensor({ (size_t)batch_size, 10 });
		labels->initialise();
		cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size * 10));
	}
	else {
		labels = new tensor({ (size_t)batch_size });
		labels->initialise();
		cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size));
	}

	initialised = true;
}

int mnist_data_loader::read_int(char * buff)
{
	char b_r[4] = { buff[3], buff[2], buff[1], buff[0] };
	return *static_cast<int*>(static_cast<void*>(b_r));
}
