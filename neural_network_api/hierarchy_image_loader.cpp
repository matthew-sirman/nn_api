#include "stdafx.h"
#include "hierarchy_image_loader.h"


namespace nnet {
	hierarchy_image_loader::hierarchy_image_loader(string file_path, shape target_size, bool one_hot)
	{
		//set the parameters
		this->file_path = file_path;
		this->target_shape = target_size;
		this->one_hot = one_hot;
	}

	hierarchy_image_loader::~hierarchy_image_loader()
	{
	}

	void hierarchy_image_loader::load_data_set()
	{
		//initialise the number of classes to 0
		n_classes = 0;
		//initialise the number of items to 0
		n_items = 0;

		//iterate through the folder
		for (directory_iterator dir_iter(file_path); dir_iter != directory_iterator(); dir_iter++) {
			//get the name of each subfolder
			string class_fldr = dir_iter->path().filename().string();

			//loop through each subfolder
			for (directory_iterator class_iter(file_path + "/" + class_fldr); class_iter != directory_iterator(); class_iter++) {
				//get the name of each image
				string img_name = class_iter->path().filename().string();

				//add the image name to the vector of pairings
				//n_classes represents the class this image belongs to (as the current number of classes
				//is the current class index, starting with 0)
				dataset_files.push_back(pair<int, string>(n_classes, class_fldr + "/" + img_name));

				//increment the number of items
				n_items++;
			}

			//increment the number of classes
			n_classes++;
		}
		//shuffle the dataset, as otherwise it would be in perfect order which would be bad for training
		srand(time(NULL));
		random_shuffle(dataset_files.begin(), dataset_files.end());
	}

	void hierarchy_image_loader::next_batch()
	{
		//set the data buffer to 0
		memset(data_buffer, 0, sizeof(float) * batch_size * target_shape.size());

		//set the label buffers to 0
		if (one_hot) {
			cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * n_classes * batch_size));
			memset(onehot_labels, 0, sizeof(float) * n_classes * batch_size);
		}
		else {
			cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size));
			memset(index_labels, 0, sizeof(float) * batch_size);
		}

		//set the current batch to the batch size
		size_t current_batch_size = batch_size;

		//if the number of remaining items is less than the batch size,
		//set the abtch size to the remainder
		if (n_items - load_index < batch_size) {
			current_batch_size = n_items - load_index;
		}

		//set the buffer offset to 0
		size_t offset = 0;

		for (int ld = 0; ld < current_batch_size; ld++) {
			//load an image from dataset_files array

			//cache the current element
			pair<int, string> element = dataset_files[load_index];

			string file_str = file_path + "/" + element.second;

			//read the image from file
			Mat img_mat = imread(file_str);

			//reformat loaded image and write it into the data buffer
			reformat_image(img_mat, data_buffer, offset);

			//increment the offset at which the image is written to the data buffer
			offset += target_shape.size();

			//set the label buffer to whatever class is currently loaded
			if (one_hot) {
				onehot_labels[ld * n_classes + element.first] = 1.0;
			}
			else {
				index_labels[ld] = element.first;
			}

			//increment the load index
			load_index++;
		}

		//copy the data buffer into the data tensor
		cuda_safe_call(cudaMemcpy(data->get_dev_pointer(), data_buffer, sizeof(float) * batch_size * target_shape.size(), cudaMemcpyHostToDevice));

		//copy the label buffer into the label tensor
		if (one_hot) {
			cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), onehot_labels, sizeof(float) * n_classes * batch_size, cudaMemcpyHostToDevice));
		}
		else {
			cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), index_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice));
		}
	}

	tensor* hierarchy_image_loader::get_next_batch()
	{
		return data;
	}

	tensor* hierarchy_image_loader::get_next_batch_labels()
	{
		return labels;
	}

	void hierarchy_image_loader::reset_iterator()
	{
		load_index = 0;
		//reshuffle the dataset to improve performance
		random_shuffle(dataset_files.begin(), dataset_files.end());
	}

	void hierarchy_image_loader::initialise(size_t batch_size)
	{
		//if already initialised abort and return
		if (initialised)
			return;

		//set the local batch size to the input batch size
		this->batch_size = batch_size;

		//declare the data buffer
		data_buffer = (float*)malloc(sizeof(float) * batch_size * target_shape.size());

		//setup the data tensor with the batch size in the first dimension and the target size in the second
		//dimensions
		data = new tensor({ batch_size, target_shape.width, target_shape.height, target_shape.depth });
		data->initialise();

		//declare the label buffers
		index_labels = (float*)malloc(sizeof(float) * batch_size);
		onehot_labels = (float*)malloc(sizeof(float) * n_classes * batch_size);

		//setup the label tensor
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

		//set the initialised flag to true
		initialised = true;
	}

	void hierarchy_image_loader::close()
	{
		//if not initialised return
		if (!initialised)
			return;

		//release all the memory
		free(index_labels);
		free(onehot_labels);
		free(data_buffer);
		data->uninitialise();
		labels->uninitialise();

		//set the initialised flag to false
		initialised = false;
	}

	void hierarchy_image_loader::reformat_image(Mat img, float* buffer, size_t offset)
	{
		//get the start rows and cols from the raw image
		int r_w = img.cols;
		int r_h = img.rows;

		//get the percentage the image fills of the target shape
		float f_x = r_w / (float)target_shape.width;
		float f_y = r_h / (float)target_shape.height;

		//get the maximum scale dimension
		float f = max(f_x, f_y);

		//get the new width and the new height
		int n_w = max((int)min((float)target_shape.width, r_w / f), 1);
		int n_h = max((int)min((float)target_shape.height, r_h / f), 1);

		//declare the scaled image
		Mat scl_img;

		//resize the image into the scaled image
		resize(img, scl_img, Size(n_w, n_h));

		//declare the float matrix
		Mat fmt_img;

		//copy the image into the float matrix and scale it respectively
		scl_img.convertTo(fmt_img, CV_32F, 1 / 255.0);

		//get the target rectangle from the target
		Rect tgt_rect = Rect(0, 0, n_w, n_h);

		//target matrix of the correct output size
		Mat target = Mat(target_shape.height, target_shape.width, CV_32FC((int)target_shape.depth));
		//copy the float image into the output target image
		fmt_img.copyTo(target(tgt_rect));

		//split the image channels
		//OpenCV uses RGB RGB RGB formatting for each pixel
		//this api uses RRR GGG BBB formatting for each pixel,
		//so it is required to split the channels
		vector<Mat> channels(target_shape.depth);
		split(target, channels);

		//set the channel offset to start with the buffer offset
		size_t c_off = offset;
		for (Mat channel : channels) {
			//copy this channel into the buffer
			memcpy(&buffer[c_off], (float*)channel.data, sizeof(float) * channel.rows * channel.cols);
			//increase the offset
			c_off += channel.rows * channel.cols;
		}
	}
}