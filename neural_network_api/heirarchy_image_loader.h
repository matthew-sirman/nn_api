#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>
#include <boost/filesystem.hpp>

#include "batch_iterator.h"
#include "shape.h"

using namespace boost::filesystem;
using namespace cv;

class NN_LIB_API heirarchy_image_loader : public batch_iterator
{
public:
	heirarchy_image_loader(string file_path, shape target_shape, bool one_hot = true);
	~heirarchy_image_loader();

	void load_data_set();
	void next_batch() override;
	tensor * get_next_batch() override;
	tensor * get_next_batch_labels() override;
	void reset_iterator() override;
	void initialise(size_t batch_size) override;
	void close() override;

	void reformat_image(Mat img, float * buffer, size_t offset);
private:
	string file_path;
	vector<pair<int, string>> dataset_files;

	shape target_shape;

	tensor * data;
	tensor * labels;

	float * data_buffer;

	float * index_labels;
	float * onehot_labels;

	size_t load_index = 0;

	bool one_hot;
};

