// neural_network_api.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

//TEMP
#include <stdio.h>
#include "network_model.h"
#include "cifar_data_loader.h"
#include "mnist_data_loader.h"
#include <opencv2\opencv.hpp>

using namespace cv;

int main() {
	/*cifar_10_data_loader loader = cifar_10_data_loader("D:\\Users\\Matthew\\Neural Networks\\datasets\\cifar10\\train", false);
	loader.load_data_set();

	loader.initialise(128);

	int n = 18;

	for (int i = 0; i < 15; i++)
		loader.next_batch();

	float * labels = (float *)malloc(sizeof(float) * 1);
	cudaMemcpy(labels, &loader.get_next_batch_labels()->get_dev_pointer()[n], sizeof(float) * 1, cudaMemcpyDeviceToHost);

	float * img = (float *)malloc(sizeof(float) * 32 * 32 * 3);
	cudaMemcpy(img, &loader.get_next_batch()->get_dev_pointer()[n * 32 * 32 * 3], sizeof(float) * 32 * 32 * 3, cudaMemcpyDeviceToHost);

	/*for (int k = 0; k < 3; k++) {
		printf("Layer %d\n\n", k);
		for (int m = 0; m < 32; m++) {
			for (int n = 0; n < 32; n++) {
				printf("%.2f ", img[k * 32 * 32 + m * 32 + n]);
			}
			printf("\n");
		}
		printf("\n");
	}/**/

	/*float * trans_img = (float *)malloc(sizeof(float) * 32 * 32 * 3);

	for (int k = 0; k < 3; k++) {
		for (int m = 0; m < 32; m++) {
			for (int n = 0; n < 32; n++) {
				trans_img[m * 32 * 3 + n * 3 + k] = img[k * 32 * 32 + m * 32 + n];
			}
		}
	}

	Mat m = Mat(32, 32, CV_32FC3, trans_img);
	Mat big;

	resize(m, big, Size(32 * 8, 32 * 8), 0, 0, INTER_CUBIC);

	imshow("Big", big);
	imshow("Original", m);

	printf("Label: %f\n", labels[0]);

	waitKey(0);

	/*for (int i = 0; i < 128; i++) {
		printf("Label[%d] = %f\n", i, labels[i]);
	}*/
	//mnist_data_loader loader = mnist_data_loader("D:\\Users\\Matthew\\Neural Networks\\datasets\\cifar10");
	//loader.load_data_set("data_batch_1.bin");
	return 0;
}