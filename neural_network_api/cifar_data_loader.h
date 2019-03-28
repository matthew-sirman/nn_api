#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>

#include "batch_iterator.h"
#include "instructions_kernel.h"

/*
FORMAT:
1 byte label (or 2 bytes for CIFAR-100)
3072 byte data
*/

using namespace std;
using namespace boost::filesystem;

enum NN_LIB_API cifar_dataset {
	CIFAR_10,
	CIFAR_100_F,
	CIFAR_100_C
};

template <cifar_dataset cifar, size_t CLASSES>
class cifar_data_loader : public batch_iterator
{
public:
	cifar_data_loader(string file_path, bool one_hot = true) {
		this->file_path = file_path;
		this->one_hot = one_hot;
		this->n_classes = CLASSES;

		switch (cifar) {
		case CIFAR_10:
			label_depth = 1;
			break;
		case CIFAR_100_F:
			label_depth = 2;
			break;
		case CIFAR_100_C:
			label_depth = 2;
			break;
		}
	}

	~cifar_data_loader() {};

	void load_data_set() {
		n_items = 0;

		for (directory_iterator dir_iter(file_path); dir_iter != directory_iterator(); dir_iter++) {
			string file_str = dir_iter->path().filename().string();
			file_names.push_back(file_str);

			std::ifstream data_stream = std::ifstream(file_path + "\\" + file_str, ios::binary);
			if (data_stream.fail()) {
				throw new exception((string("Failed to load file ") + file_str).c_str());
			}

			data_stream.seekg(0, data_stream.end);
			size_t length = data_stream.tellg();
			data_stream.seekg(0, data_stream.beg);

			n_items += length / (n_rows * n_cols * depth + label_depth);
		}
	}

	void close() override {
		if (!initialised)
			return;
		free(data_buffer);
		cuda_safe_call(cudaFree(d_data_buffer));
		//free(label_buffer);
		free(onehot_labels);
		free(index_labels);
		data->uninitialise();
		labels->uninitialise();
		initialised = false;
	}

	void next_batch() override {
		std::ifstream data_stream = std::ifstream(file_path + "\\" + file_names[__file_index], ios::binary);

		data_stream.seekg(0, data_stream.end);
		size_t total_file_length = data_stream.tellg();
		size_t length = total_file_length - (d_s_index - file_len_pos);
		data_stream.seekg(d_s_index - file_len_pos);

		size_t batch_load_size = batch_size * (n_rows * n_cols * depth + label_depth);

		size_t current_batch_size = batch_size;

		if (length >= batch_load_size) {
			if (length == batch_load_size) {
				__file_index++;
				file_len_pos += total_file_length;
			}
			length = batch_load_size;

			//load entire batch as it is all located in this file.

			data_stream.read(data_buffer, length);
		}
		else {
			data_stream.read(data_buffer, length);

			__file_index++;
			if (__file_index < file_names.size()) {
				std::ifstream next_data_stream = std::ifstream(file_path + "\\" + file_names[__file_index], ios::binary);

				next_data_stream.seekg(0, next_data_stream.end);
				size_t next_len = next_data_stream.tellg();
				next_data_stream.seekg(0, next_data_stream.beg);

				file_len_pos += total_file_length;

				if (next_len >= batch_load_size - length) {
					next_data_stream.read(&data_buffer[length], batch_load_size - length);
				}

				next_data_stream.close();
			}
			else {
				memset(&data_buffer[length], 0, batch_load_size - length);
				current_batch_size = n_items % batch_size;
			}
		}

		data->set_shape({ current_batch_size, n_rows, n_cols, depth });
		if (one_hot) {
			labels->set_shape({ current_batch_size, n_classes });
		}
		else {
			labels->set_shape({ current_batch_size });
		}

		memset(onehot_labels, 0, sizeof(float) * batch_size * n_classes);
		memset(index_labels, 0, sizeof(float) * batch_size);
		for (int elem = 0; elem < batch_size; elem++) {
			if (elem < current_batch_size) {
				int label;
				switch (cifar) {
				case CIFAR_10:
					label = data_buffer[elem * (n_rows * n_cols * depth + label_depth)];
					break;
				case CIFAR_100_F:
					label = data_buffer[elem * (n_rows * n_cols * depth + label_depth) + 1];
					break;
				case CIFAR_100_C:
					label = data_buffer[elem * (n_rows * n_cols * depth + label_depth)];
					break;
				}
				cuda_safe_call(cudaMemcpy(
					&d_data_buffer[elem * n_rows * n_cols * depth],
					&data_buffer[elem * (n_rows * n_cols * depth + label_depth) + label_depth],
					sizeof(byte) * n_rows * n_cols * depth,
					cudaMemcpyHostToDevice
				));

				if (one_hot) {
					onehot_labels[elem * n_classes + label] = 1;
				}
				else {
					index_labels[elem] = label;
				}
			}
			else {
				//label_buffer[elem] = 0;
				cuda_safe_call(cudaMemset(&d_data_buffer[elem * n_rows * n_cols * depth], 0, sizeof(byte) * n_rows * n_cols * depth));
			}
		}

		scalar_matrix_multiply_b(d_data_buffer, data->get_dev_pointer(), 1.0 / 255.0, batch_size * n_rows * n_cols * depth);

		if (one_hot) {
			cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), onehot_labels, sizeof(float) * n_classes * batch_size, cudaMemcpyHostToDevice));
		}
		else {
			cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), index_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice));
		}

		data_stream.close();
		d_s_index += length;
	}

	tensor * get_next_batch() override {
		return data;
	}

	tensor * get_next_batch_labels() override {
		return labels;
	}

	void reset_iterator() override {
		d_s_index = 0;
		file_len_pos = 0;
		__file_index = 0;
	}

	void initialise(size_t batch_size) override {
		if (initialised)
			return;

		this->batch_size = batch_size;

		data_buffer = (char *)malloc(sizeof(char) * (n_rows * n_cols * depth + label_depth) * batch_size);
		cuda_safe_call(cudaMallocManaged(&d_data_buffer, sizeof(byte) * n_rows * n_cols * depth * batch_size));

		data = new tensor({ batch_size, n_rows, n_cols, depth });
		data->initialise();

		onehot_labels = (float *)malloc(sizeof(float) * n_classes * batch_size);
		index_labels = (float *)malloc(sizeof(float) * batch_size);

		if (one_hot) {
			labels = new tensor({ batch_size, n_classes });
			labels->initialise();
			cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * n_classes * batch_size));
		}
		else {
			labels = new tensor({ batch_size });
			labels->initialise();
			cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size));
		}

		initialised = true;
	}

private:
	string file_path;
	vector<string> file_names;

	//this is a dataset specific loader so these can be const
	const size_t n_rows = 32;
	const size_t n_cols = 32;
	const size_t depth = 3;
	size_t label_depth;

	tensor * data;
	tensor * labels;

	bool one_hot = true;

	char * data_buffer;
	byte * d_data_buffer;

	float * index_labels;
	float * onehot_labels;

	size_t file_len_pos = 0;
	size_t d_s_index = 0;

	int __file_index = 0;
};

typedef cifar_data_loader<CIFAR_10, 10> cifar_10_data_loader;
typedef cifar_data_loader<CIFAR_100_F, 100> cifar_100_fine_data_loader;
typedef cifar_data_loader<CIFAR_100_C, 10> cifar_100_coarse_data_loader;