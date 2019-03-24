#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <iostream>
#include <fstream>
#include <string>

#include "batch_iterator.h"
#include "instructions_kernel.h"

using namespace std;

class NN_LIB_API mnist_data_loader : public batch_iterator
{
public:
	mnist_data_loader(string file_path, bool one_hot = true, int classes = 10);
	mnist_data_loader(string file_path, size_t batch_size, bool one_hot = true, int classes = 10);
	~mnist_data_loader();

	void load_data_set(string file_name);
	void load_data_set_labels(string file_name);

	void close() override;

	tensor * get_next_batch() override;
	tensor * get_next_batch_labels() override;
	void reset_iterator() override;
	void initialise(size_t batch_size) override;

private:
	static int read_int(char * buff);

	string file_path;
	string d_file_name, l_file_name;

	bool one_hot = true;

	tensor * data;
	tensor * labels;

	int magic_num;
	int magic_num_labels;

	size_t n_rows;
	size_t n_cols;

	char * data_buffer;
	byte * d_data_buffer;

	char * label_buffer;
	float * onehot_labels;
	float * index_labels;

	size_t d_s_index = 0;
	size_t l_s_index = 0;
};

