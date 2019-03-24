#include "stdafx.h"
#include "heirarchy_image_loader.h"


heirarchy_image_loader::heirarchy_image_loader(string file_path, shape target_size, bool one_hot)
{
	this->file_path = file_path;
	this->target_shape = target_size;
	this->one_hot = one_hot;
}

heirarchy_image_loader::~heirarchy_image_loader()
{
}

void heirarchy_image_loader::load_data_set()
{
	n_classes = 0;
	n_items = 0;
	for (directory_iterator dir_iter(file_path); dir_iter != directory_iterator(); dir_iter++) {
		string class_fldr = dir_iter->path().filename().string();

		for (directory_iterator class_iter(file_path + "/" + class_fldr); class_iter != directory_iterator(); class_iter++) {
			string img_name = class_iter->path().filename().string();
			dataset_files.push_back(pair<int, string>(n_classes, class_fldr + "/" + img_name));

			n_items++;
		}

		n_classes++;
	}
	srand(time(NULL));
	random_shuffle(dataset_files.begin(), dataset_files.end());
}

void heirarchy_image_loader::next_batch()
{
	memset(data_buffer, 0, sizeof(float) * batch_size * target_shape.size());

	if (one_hot) {
		cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * n_classes * batch_size));
		memset(onehot_labels, 0, sizeof(float) * n_classes * batch_size);
	}
	else {
		cuda_safe_call(cudaMemset(labels->get_dev_pointer(), 0, sizeof(float) * batch_size));
		memset(index_labels, 0, sizeof(float) * batch_size);
	}

	size_t current_batch_size = batch_size;
	if (n_items - load_index < batch_size) {
		current_batch_size = n_items - load_index;
	}

	size_t offset = 0;

	for (int ld = 0; ld < current_batch_size; ld++) {
		//load an image from dataset_files array

		pair<int, string> element = dataset_files[load_index];

		string file_str = file_path + "/" + element.second;

		Mat img_mat = imread(file_str);

		reformat_image(img_mat, data_buffer, offset);
		offset += target_shape.size();

		if (one_hot) {
			onehot_labels[ld * n_classes + element.first] = 1.0;
		}
		else {
			index_labels[ld] = element.first;
		}

		load_index++;
	}

	cuda_safe_call(cudaMemcpy(data->get_dev_pointer(), data_buffer, sizeof(float) * batch_size * target_shape.size(), cudaMemcpyHostToDevice));

	if (one_hot) {
		cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), onehot_labels, sizeof(float) * n_classes * batch_size, cudaMemcpyHostToDevice));
	}
	else {
		cuda_safe_call(cudaMemcpy(labels->get_dev_pointer(), index_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice));
	}
}

tensor * heirarchy_image_loader::get_next_batch()
{
	return data;
}

tensor * heirarchy_image_loader::get_next_batch_labels()
{
	return labels;
}

void heirarchy_image_loader::reset_iterator()
{
	load_index = 0;
	random_shuffle(dataset_files.begin(), dataset_files.end());
}

void heirarchy_image_loader::initialise(size_t batch_size)
{
	if (initialised)
		return;

	this->batch_size = batch_size;


	data_buffer = (float *)malloc(sizeof(float) * batch_size * target_shape.size());
	data = new tensor({ batch_size, target_shape.width, target_shape.height, target_shape.depth });
	data->initialise();

	index_labels = (float *)malloc(sizeof(float) * batch_size);
	onehot_labels = (float *)malloc(sizeof(float) * n_classes * batch_size);

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

void heirarchy_image_loader::close()
{
	if (!initialised)
		return;

	free(index_labels);
	free(onehot_labels);
	free(data_buffer);
	data->uninitialise();
	labels->uninitialise();

	initialised = false;
}

void heirarchy_image_loader::reformat_image(Mat img, float * buffer, size_t offset)
{
	int r_w = img.cols;
	int r_h = img.rows;

	float f_x = r_w / (float)target_shape.width;
	float f_y = r_h / (float)target_shape.height;

	float f = max(f_x, f_y);

	int n_w = max((int)min((float)target_shape.width, r_w / f), 1);
	int n_h = max((int)min((float)target_shape.height, r_h / f), 1);

	Mat scl_img;
	resize(img, scl_img, Size(n_w, n_h));

	Mat fmt_img;
	scl_img.convertTo(fmt_img, CV_32F, 1/255.0);
	
	Rect tgt_rect = Rect(0, 0, n_w, n_h);

	Mat target = Mat(target_shape.height, target_shape.width, CV_32FC((int)target_shape.depth));
	fmt_img.copyTo(target(tgt_rect));

	vector<Mat> channels(target_shape.depth);
	split(target, channels);

	size_t c_off = offset;
	for (Mat channel : channels) {
		memcpy(&buffer[c_off], (float *)channel.data, sizeof(float) * channel.rows * channel.cols);
		c_off += channel.rows * channel.cols;
	}
}
