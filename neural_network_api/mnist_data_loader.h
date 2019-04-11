#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "batch_iterator.h"
#include "instructions_kernel.h"

using namespace std;

namespace nnet {
	//MNIST Data Loader
	//Loads information in the MNIST format
	//This class can be used for the MNIST, EMNIST, FASION MNIST datasets, and may be used for
	//other datasets following the same format.
	//FORMAT OF MNIST:
	//Images:
	//Metadata specifying magic number, number of items, rows and colums
	//Rows by cols pixel grayscale images stored in binary sequence
	//Default for MNIST: 28x28 pixels
	//Labels:
	//Metadata specifying magic number and number of items
	//1 byte labels for each image
	//
	//The image file and label file must have the same number of items, and
	//the sequences must be the corresponding
	class mnist_data_loader : public batch_iterator
	{
	public:
		//Constructor specifying the folder path, whether the data should be loaded
		//in one-hot encoding and the number of classes
		mnist_data_loader(string file_path, bool one_hot = true, int classes = 10);

		//Destructor
		~mnist_data_loader();

		//Load Data Set
		//Specify the file name of the dataset.
		//The file must be in the folder specified upon construction
		void load_data_set(string file_name);

		//Load Data Set Labels
		//Specify the file name of the dataset labels.
		//The file must be in the folder specified upon construction
		void load_data_set_labels(string file_name);

		//Close
		//Close the dataset after use
		void close() override;

		//API FUNCTION
		//Get Next Batch
		//Returns the next batch of images from the file
		tensor* get_next_batch() override;

		//API FUNCTION
		//Get Next Batch Labels
		//Returns the next batch of labels from the file
		tensor* get_next_batch_labels() override;

		//API FUNCTION
		//Reset Iterator
		//Called after every epoch to reset the iterator to the start of the dataset
		void reset_iterator() override;

		//API FUNCTION
		//Initialise
		//Initialises the iterator by specifying the batch size
		void initialise(size_t batch_size) override;

		//Load Size
		//Get the input shape for the dataset (by default 28x28x1)
		shape load_size() { return shape(n_rows, n_cols); }

	private:
		//convert a binary stream buffer to an integer
		static int read_int(char* buff);

		//the file path which contains the dataset files
		string file_path;

		//the file name for the dataset and its labels
		string d_file_name, l_file_name;

		//flag to indicate if the dataset should be loaded with one hot encoding
		bool one_hot = true;

		//tensor to store the data (images for MNIST)
		tensor* data;
		//tensor to store labels for the data
		tensor* labels;

		//save the magic numbers from each file
		//(not currently actually used)
		int magic_num;
		int magic_num_labels;

		//image rows and columns
		size_t n_rows;
		size_t n_cols;

		//buffers for loading and passing data
		char* data_buffer;
		byte* d_data_buffer;

		//buffers for loading and passing labels
		char* label_buffer;
		float* onehot_labels;
		float* index_labels;

		//variables to store whereabouts in which file we currently are
		size_t d_s_index = 0;
		size_t l_s_index = 0;
	};

}