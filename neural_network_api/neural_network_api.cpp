// neural_network_api.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

//TEMP
#include <stdio.h>
#include <fstream>
#include "network_model.h"
#include "cifar_data_loader.h"
#include <opencv2\opencv.hpp>
#include <boost/lexical_cast.hpp>

//using namespace cv;
using namespace std;

int main() {

	/*int b_sz = 4;

	float * in = (float *)malloc(sizeof(float) * 28 * 28 * 8 * b_sz);
	float * out = (float *)malloc(sizeof(float) * 32 * 32 * 3 * b_sz);
	float * fltr = (float *)malloc(sizeof(float) * 5 * 5 * 3 * 8);

	float * d_in;
	float * d_out;
	float * d_fltr;

	cudaMallocManaged(&d_in, sizeof(float) * 28 * 28 * 8 * b_sz);
	cudaMallocManaged(&d_out, sizeof(float) * 32 * 32 * 3 * b_sz);
	cudaMallocManaged(&d_fltr, sizeof(float) * 5 * 5 * 3 * 8);

	std::fill(&in[28 * 28 * 0], &in[28 * 28 * 8 * 2], 0.5);
	std::fill(&in[28 * 28 * 8 * 2], &in[28 * 28 * 8 * b_sz], 0.6);
	std::fill(&out[32 * 32 * 0], &out[32 * 32 * 3 * b_sz], 0.5);

	cudaMemcpy(d_in, in, sizeof(float) * 28 * 28 * 8 * b_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, sizeof(float) * 32 * 32 * 3 * b_sz, cudaMemcpyHostToDevice);

	filter_convolve_2d_derivative(
		d_out,
		d_in,
		d_fltr,
		shape(32, 32, 3),
		shape(28, 28, 8),
		shape(5, 5, 3),
		shape(0, 0),
		b_sz
	);

	cudaMemcpy(fltr, d_fltr, sizeof(float) * 5 * 5 * 3 * 8, cudaMemcpyDeviceToHost);

	for (int f = 0; f < 8; f++) {
		printf("Filter %d:\n\n", f);
		for (int k = 0; k < 3; k++) {
			printf("Layer %d:\n", k);
			for (int m = 0; m < 5; m++) {
				for (int n = 0; n < 5; n++) {
					printf("%.2f ", fltr[f * 5 * 5 * 3 + k * 5 * 5 + m * 5 + n]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}

	return 0;*/

	network_model model;
	model.entry(shape(32, 32, 3));
	//tensor filter = tensor::random({ 5, 5, 3, 8 }, -0.1, 0.1);
	model.conv2d(shape(5, 5, 3), 8);

	//float * fltr = (float *)malloc(sizeof(float) * 5 * 5 * 3 * 8);
	//cudaMemcpy(fltr, filter.get_dev_pointer(), sizeof(float) * 5 * 5 * 3 * 8, cudaMemcpyDeviceToHost);
	//"D:\Users\Matthew\Neural Networks\TEST_params"

	/*std::ofstream out_str = std::ofstream("D:/Users/Matthew/Neural Networks/TEST_params/filter.txt");

	for (int f = 0; f < 8; f++) {
		for (int k = 0; k < 3; k++) {
			for (int m = 0; m < 5; m++) {
				for (int n = 0; n < 5; n++) {
					float f_val = filter.get_data()[f * 5 * 5 * 3 + k * 5 * 5 + m * 5 + n];
					string o_string = boost::lexical_cast<string>(f_val) + "\n";
					out_str.write(o_string.c_str(), o_string.size());
				}
			}
		}
	}

	out_str.close();/**/

	model.tanh();
	model.flatten();
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