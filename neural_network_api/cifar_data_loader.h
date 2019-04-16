#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>

#include "batch_iterator.h"
#include "instructions_kernel.h"
#include "error_handling.h"

/*
FORMAT:
1 byte label (or 2 bytes for CIFAR-100)
3072 byte data
*/

using namespace std;
using namespace boost::filesystem;

namespace nnet {
	//Cifar Dataset
	//Specifies which CIFAR style dataset to use as defaults. The loader isn't necessarily exclusive to these
	//types, however support for other datasets is not necessarily available
	enum cifar_dataset {
		CIFAR_10,
		CIFAR_100_F,
		CIFAR_100_C
	};

	//CIFAR Dataset style specific loader
	//Loads information in the CIFAR format
	//This class can be used for the CIFAR-10 and CIFAR-100 datasets, and may be used for
	//other datasets following the same format.
	//FORMAT OF CIFAR-10:
	//3073 bytes per element
	//Byte 0 -> Label byte
	//Byte 1-3073 -> Image bytes (Cols by Rows by Depth with RGB format)
	//
	//FORMAT OF CIFAR-100:
	//3074 bytes per element
	//Byte 0 -> Coarse label byte
	//Byte 1 -> Fine label byte
	//Byte 2-3074 -> Image bytes (Cols by Rows by Depth with RGB format)
	template <cifar_dataset cifar, size_t CLASSES, size_t WIDTH, size_t HEIGHT, size_t DEPTH>
	class cifar_data_loader : public batch_iterator
	{
	public:

		//Constructor with file_path and one_hot
		//File path should specify the folder in which the dataset is stored.
		//The reader will attempt to read every file within this folder.
		//One-hot specifies if the label tensor should be one-hot formatted or sparse.
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

		//Destructor
		~cifar_data_loader() {};

		//Loads Dataset
		//Load the dataset by working out the number of elements and all the files within
		//the path folder.
		void load_data_set() {
			//Initialise the number of items to 0
			n_items = 0;

			//Iterate through the folder specified
			for (directory_iterator dir_iter(file_path); dir_iter != directory_iterator(); dir_iter++) {
				//get the file name for the current file
				string file_str = dir_iter->path().filename().string();
				//add the file name to a list of all the files the loader should load from
				file_names.push_back(file_str);

				//open the current file
				std::ifstream data_stream = std::ifstream(file_path + "\\" + file_str, ios::binary);

				//if the file open fails for any reason throw an exception
				//POSSIBLY REMOVE THIS AND JUST IGNORE THE FILE?
				ERR_ASSERT(data_stream.fail(), "Failed to load file: " << file_str);

				//Get the length of the file
				data_stream.seekg(0, data_stream.end);
				size_t length = data_stream.tellg();
				data_stream.seekg(0, data_stream.beg);

				//close the file as we are finished with it for now
				data_stream.close();

				//add the length divided by the size of each element to the number of items
				n_items += length / (WIDTH * HEIGHT * DEPTH + label_depth);
			}
		}

		//Close
		//Closes the dataset loader by dereferncing memory and uninitialising tensors
		void close() override {
			//if the loader isn't initialised it can't be closed so abort and return
			if (!initialised)
				return;
			//free all data used by the loader
			free(data_buffer);
			cuda_safe_call(cudaFree(d_data_buffer));
			free(onehot_labels);
			free(index_labels);
			//uninitialise tensors
			data->uninitialise();
			labels->uninitialise();
			//set initialsed to false to indicate that the loader was uninitialised
			initialised = false;
		}

		//API FUNCTION
		//Next Batch
		//Calculates the next batch by simultaneously loading images and labels
		void next_batch() override {
			//open the current file (as it is batched in files and the loader) as a binary file
			std::ifstream data_stream = std::ifstream(file_path + "\\" + file_names[__file_index], ios::binary);

			//find the total file length and subtract the current index through which the iterator is
			//(as many batches will start part way through a file)
			data_stream.seekg(0, data_stream.end);
			size_t total_file_length = data_stream.tellg();
			size_t length = total_file_length - (d_s_index - file_len_pos);
			data_stream.seekg(d_s_index - file_len_pos);

			//set the batch loading size assuming an entire batch will be used
			size_t batch_load_size = batch_size * (WIDTH * HEIGHT * DEPTH + label_depth);

			//initialise the current batch size to the batch size
			size_t current_batch_size = batch_size;

			//if the file contains the entire batch
			if (length >= batch_load_size) {
				//if the end of the file and the end of the batch line up perfectly the file index must be incremented
				//so the next batch is loaded from the next file
				if (length == batch_load_size) {
					__file_index++;
					file_len_pos += total_file_length;
				}
				//overwrite the length such that the size of the batch is loaded from the file
				length = batch_load_size;

				//load entire batch into buffer as it is all located in this file.
				data_stream.read(data_buffer, length);
			}
			//if only part of the batch is located in this file
			else {
				//read the part in this file (which must be the rest of the file, i.e. the length)
				data_stream.read(data_buffer, length);

				//move to the next file
				__file_index++;
				//check that the file is within the range of files
				if (__file_index < file_names.size()) {
					//open the next file
					std::ifstream next_data_stream = std::ifstream(file_path + "\\" + file_names[__file_index], ios::binary);

					//get the total length of the new file
					//no separate offset length variable is required because we are starting at the beginning
					//of this file, and assuming the batch size is lower than the length of an entire file
					//(otherwise a more complex recursive algorithm would be required, which would never
					//be necessary as it would not be optimal to use a batch size large enough)
					next_data_stream.seekg(0, next_data_stream.end);
					size_t next_len = next_data_stream.tellg();
					next_data_stream.seekg(0, next_data_stream.beg);

					//add the length of the new file to the position variable (for offsetting when loading the file)
					file_len_pos += total_file_length;

					//check that the file is big enough for the batch size to be loaded across two files
					if (next_len >= batch_load_size - length) {
						//read in th remainder of the batch from the next file at the end of the buffer
						next_data_stream.read(&data_buffer[length], batch_load_size - length);
					}

					//close the next stream file as data is already loaded
					next_data_stream.close();
				}
				else {
					//if the batch goes over the end of the last file set the remainder of the buffer to 0
					memset(&data_buffer[length], 0, batch_load_size - length);
					//reduce the current batch size (as it must be smaller)
					current_batch_size = n_items % batch_size;
				}
			}

			//set the shape of the data tensor
			data->set_shape({ current_batch_size, WIDTH, HEIGHT, DEPTH });
			//set the shape of the label tensor
			if (one_hot) {
				labels->set_shape({ current_batch_size, n_classes });
			}
			else {
				labels->set_shape({ current_batch_size });
			}

			//set the label buffers to 0s
			memset(onehot_labels, 0, sizeof(float) * batch_size * n_classes);
			memset(index_labels, 0, sizeof(float) * batch_size);

			//loop through each element in the batch
			for (int elem = 0; elem < batch_size; elem++) {
				//if the element has been loaded
				if (elem < current_batch_size) {
					//get the label from the data buffer depending on which data model was used
					int label;
					switch (cifar) {
					case CIFAR_10:
						label = data_buffer[elem * (WIDTH * HEIGHT * DEPTH + label_depth)];
						break;
					case CIFAR_100_F:
						label = data_buffer[elem * (WIDTH * HEIGHT * DEPTH + label_depth) + 1];
						break;
					case CIFAR_100_C:
						label = data_buffer[elem * (WIDTH * HEIGHT * DEPTH + label_depth)];
						break;
					}

					//copy the data buffer into device (GPU) memory
					//as this load is staggered another buffer is used to transfer the data
					//rather than writing straght to the tensor
					cuda_safe_call(cudaMemcpy(
						&d_data_buffer[elem * WIDTH * HEIGHT * DEPTH],
						&data_buffer[elem * (WIDTH * HEIGHT * DEPTH + label_depth) + label_depth],
						sizeof(byte) * WIDTH * HEIGHT * DEPTH,
						cudaMemcpyHostToDevice
					));

					//update the label buffer with the label information
					if (one_hot) {
						onehot_labels[elem * n_classes + label] = 1;
					}
					else {
						index_labels[elem] = label;
					}
				}
				else {
					//if the element is outside the range of the current batch, set the memory to 0
					cuda_safe_call(cudaMemset(&d_data_buffer[elem * WIDTH * HEIGHT * DEPTH], 0, sizeof(byte) * WIDTH * HEIGHT * DEPTH));
				}
			}

			//normalise the loaded data to be from 0-1 rather than 0-255. Also write it out as a float array instead of byte array
			//into the tensor's device data
			scalar_matrix_multiply_b(d_data_buffer, data->get_dev_pointer(), 1.0 / 255.0, batch_size * WIDTH * HEIGHT * DEPTH);

			//copy the label buffer into the label tensor's device data
			if (one_hot) {
				cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), onehot_labels, sizeof(float) * n_classes * batch_size, cudaMemcpyHostToDevice));
			}
			else {
				cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), index_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice));
			}

			//close the datastream
			data_stream.close();
			//increment the batching index (how far through the entire dataset we are after loading this batch)
			d_s_index += length;
		}

		//API FUNCTION
		//Get Next Batch
		//Returns the data for the batch loaded by next_batch()
		tensor* get_next_batch() override {
			return data;
		}

		//API FUNCTION
		//Get Next Batch Labels
		//Returns the labels for the batch loaded by next_batch()
		tensor* get_next_batch_labels() override {
			return labels;
		}

		//API FUNCTION
		//Reset Iterator
		//Resets the loading iterator to 0 after each epoch to start loading from the
		//beginning
		void reset_iterator() override {
			d_s_index = 0;
			file_len_pos = 0;
			__file_index = 0;
		}

		//Initialise
		//Sets the batch size and sets up the memory pointers for the loader
		//This method is called by default by the API when training.
		void initialise(size_t batch_size) override {
			//if not already initialised
			if (initialised)
				return;

			//set the local batch size to the batch size sent in
			this->batch_size = batch_size;

			//allocates a data buffer in host (CPU) memory to load from
			data_buffer = (char*)malloc(sizeof(char) * (WIDTH * HEIGHT * DEPTH + label_depth) * batch_size);

			//allocates a passing central buffer in device to memory
			cuda_safe_call(cudaMallocManaged(&d_data_buffer, sizeof(byte) * WIDTH * HEIGHT * DEPTH * batch_size));

			//sets up the tensor for the data which will have the batch size in the first element and each image in the second
			//three dimensions
			data = new tensor({ batch_size, WIDTH, HEIGHT, DEPTH });
			data->initialise();

			//allocate the label buffers 
			onehot_labels = (float*)malloc(sizeof(float) * n_classes * batch_size);
			index_labels = (float*)malloc(sizeof(float) * batch_size);

			//setup the label tensor and initialise the tensor
			//set the tensor memory to zero (this could be instead done with the "zeros"
			//initialiser, maybe change)
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

			//set initialised flag to true
			initialised = true;
		}

		//Load Size
		//Returns the loading size for the data (Width by Height by Depth)
		shape load_size() {
			return shape(WIDTH, HEIGHT, DEPTH);
		}

	private:
		//main file path to the dataset directory
		string file_path;
		//vector of all the file names within the directory
		vector<string> file_names;

		//the depth of the label section of each element in bytes
		size_t label_depth;

		//tensor to store the data (images for CIFAR)
		tensor* data;
		//tensor to store labels for the data
		tensor* labels;

		//flag to indicate whether the dataset is one-hot encoded or not
		bool one_hot = true;

		//buffers for loading and passing data
		char* data_buffer;
		byte* d_data_buffer;

		//buffers for loading and passing labels
		float* index_labels;
		float* onehot_labels;

		//variables to store whereabouts in which file we currently are
		size_t file_len_pos = 0;
		size_t d_s_index = 0;

		int __file_index = 0;
	};

	//specific template contstructions for different load formats

	//Dataset Loader for standard CIFAR-10
	//Each image will be loaded as 32x32 in RGB format
	//Uses 10 classes:
	//-airplane
	//-automobile
	//-bird
	//-cat
	//-deer
	//-dog
	//-frog 
	//-horse 
	//-ship 
	//-truck
	typedef cifar_data_loader<CIFAR_10, 10, 32, 32, 3> cifar_10_data_loader;

	//Dataset Loader for small CIFAR-10
	//Each image will be loaded as 24x24 in RGB format
	//Uses 10 classes:
	//-airplane
	//-automobile
	//-bird
	//-cat
	//-deer
	//-dog
	//-frog 
	//-horse 
	//-ship 
	//-truck
	typedef cifar_data_loader<CIFAR_10, 10, 24, 24, 3> cifar_10_small_data_loader;

	//Dataset Loader for fine CIFAR-100
	//Each image will be loaded as 32x32 in RGB format
	//Uses 100 classes:
	//-aquatic mammals:	beaver, dolphin, otter, seal, whale
	//-fish:	aquarium fish, flatfish, ray, shark, trout
	//-flowers:	orchids, poppies, roses, sunflowers, tulips
	//-food containers:	bottles, bowls, cans, cups, plates
	//-fruit and vegetables:	apples, mushrooms, oranges, pears, sweet peppers
	//-household electrical devices:	clock, computer keyboard, lamp, telephone, television
	//-household furniture:	bed, chair, couch, table, wardrobe
	//-insects:	bee, beetle, butterfly, caterpillar, cockroach
	//-large carnivores:	bear, leopard, lion, tiger, wolf
	//-large man - made outdoor things:	bridge, castle, house, road, skyscraper
	//-large natural outdoor scenes:	cloud, forest, mountain, plain, sea
	//-large omnivores and herbivores:	camel, cattle, chimpanzee, elephant, kangaroo
	//-medium - sized mammals:	fox, porcupine, possum, raccoon, skunk
	//-non - insect invertebrates:	crab, lobster, snail, spider, worm
	//-people:	baby, boy, girl, man, woman
	//-reptiles:	crocodile, dinosaur, lizard, snake, turtle
	//-small mammals:	hamster, mouse, rabbit, shrew, squirrel
	//-trees:	maple, oak, palm, pine, willow
	//-vehicles 1:	bicycle, bus, motorcycle, pickup truck, train
	//-vehicles 2:	lawn - mower, rocket, streetcar, tank, tractor
	typedef cifar_data_loader<CIFAR_100_F, 100, 32, 32, 3> cifar_100_fine_data_loader;

	//Dataset Loader for coarse CIFAR-100
	//Each image will be loaded as 32x32 in RGB format
	//Uses 20 classes:
	//-aquatic mammals
	//-fish
	//-flowers
	//-food containers
	//-fruit and vegetables
	//-household electrical devices
	//-household furniture
	//-insects
	//-large carnivores
	//-large man - made outdoor things
	//-large natural outdoor scenes
	//-large omnivores and herbivores
	//-medium - sized mammals
	//-non - insect invertebrates
	//-people
	//-reptiles
	//-small mammals
	//-trees
	//-vehicles 1
	//-vehicles 2
	typedef cifar_data_loader<CIFAR_100_C, 20, 32, 32, 3> cifar_100_coarse_data_loader;
}