#pragma once

#include <opencv2\opencv.hpp>
#include <boost/filesystem.hpp>

#include "batch_iterator.h"
#include "shape.h"

using namespace boost::filesystem;
using namespace cv;

namespace nnet {

	//Heirarchy Image Loader
	//Loads images from folders
	//Each folder is a class where each image in that folder is a member of that class
	class heirarchy_image_loader : public batch_iterator
	{
	public:
		//Constructor
		//Specifies the file path which contains a series of folders, each of which represents one class
		//Specifies the target shape which the images should be cast to (as different datasets may have
		//different image sizes, and not all images may be the same size)
		//Specifies whether the data should be loaded as one-hot
		heirarchy_image_loader(string file_path, shape target_shape, bool one_hot = true);

		//Destructor
		~heirarchy_image_loader();

		//Load Data Set
		//Load the dataset and work out the classes and what images are in each of them
		void load_data_set();

		//API FUNCTION
		//Next Batch
		//Calculates the next batch by loading the images and matching them with their labels
		void next_batch() override;

		//API FUNCTION
		//Get Next Batch
		//Returns the data for the batch loaded by next_batch()
		tensor* get_next_batch() override;

		//API FUNCTION
		//Get Next Batch Labels
		//Returns the labels for the batch loaded by next_batch()
		tensor* get_next_batch_labels() override;

		//API FUNCTION
		//Reset Iterator
		//Resets the loading iterator to 0 after each epoch to start loading from the
		//beginning
		void reset_iterator() override;

		//Initialise
		//Sets the batch size and sets up the memory pointers for the loader
		//This method is called by default by the API when training.
		void initialise(size_t batch_size) override;

		//Close
		//Closes the dataset loader by dereferncing memory and uninitialising tensors
		void close() override;

	private:
		//Reformat images to be the correct size and write to the buffer of images
		void reformat_image(Mat img, float* buffer, size_t offset);

		//the file path where all the folder classes are held
		string file_path;

		//vectoring of all the files in the dataset with their corresponding index
		vector<pair<int, string>> dataset_files;

		//the target shape
		shape target_shape;

		//tensor to store the images
		tensor* data;
		//tensor to store the labels
		tensor* labels;

		//buffers for loading and passing data
		float* data_buffer;

		//buffers for loading and passing labels
		float* index_labels;
		float* onehot_labels;

		//index to store which image is currently being loaded
		size_t load_index = 0;

		//flag to indicate whether the dataset is one-hot encoded or not
		bool one_hot;
	};

}