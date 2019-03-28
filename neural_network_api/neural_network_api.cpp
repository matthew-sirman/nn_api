// neural_network_api.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

//TEMP
#include <stdio.h>
#include <fstream>
#include "network_model.h"
#include "mnist_data_loader.h"
#include "cifar_data_loader.h"
#include <opencv2\opencv.hpp>
#include <boost/lexical_cast.hpp>

//using namespace cv;
using namespace std;

int main() {

	network_model model;
	model.entry(shape(32, 32, 3));

	model.conv2d(shape(5, 5, 3), 32, shape(2, 2));
	model.relu();
	model.max_pool(shape(2, 2), shape(2, 2));

	model.conv2d(shape(5, 5, 32), 32, shape(2, 2));
	model.relu();
	model.max_pool(shape(2, 2), shape(2, 2));

	model.conv2d(shape(5, 5, 32), 32, shape(2, 2));
	model.relu();
	model.max_pool(shape(2, 2), shape(2, 2));

	model.flatten();
	model.dense(512);
	model.relu();
	model.dense(10);

	model.set_output_function<softmax>();
	model.set_cost_function<softmax_cross_entropy>();
	model.set_optimiser(new adam(0.001));

	model.initialise_model(256);

	analytics logger = analytics(PER_EPOCH, HIGH);
	logger.plot();

	model.add_logger(logger);

	cifar_10_data_loader train("D:\\Users\\Matthew\\Neural Networks\\datasets\\cifar10\\train");
	train.load_data_set();

	model.train(train, 5);

	cifar_10_data_loader test("D:\\Users\\Matthew\\Neural Networks\\datasets\\cifar10\\test", false);
	test.load_data_set();

	float acc = model.get_accuracy(test) * 100;
	printf("Accuracy: %f\n", acc);

	model.uninitialise_model();
	train.close();
	test.close();

	/*heirarchy_image_loader loader = heirarchy_image_loader("D:/Users/Matthew/Neural Networks/datasets/caltech101", shape(300, 200, 3));
	loader.load_data_set();
	loader.initialise(128);
	loader.next_batch();*/
	return 0;
}