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
	cifar_10_data_loader train("D:\\Users\\Matthew\\Neural Networks\\datasets\\cifar10\\train");
	train.load_data_set();

	cifar_10_data_loader test("D:\\Users\\Matthew\\Neural Networks\\datasets\\cifar10\\test", false);
	test.load_data_set();/**/

	/*mnist_data_loader train = mnist_data_loader("D:\\Users\\Matthew\\Neural Networks\\datasets\\mnist", true, 47);
	train.load_data_set("train-images.idx3-ubyte");
	train.load_data_set_labels("train-labels.idx1-ubyte");*/

	/*mnist_data_loader test = mnist_data_loader("D:\\Users\\Matthew\\Neural Networks\\datasets\\mnist", false, 47);
	test.load_data_set("t10k-images.idx3-ubyte");
	test.load_data_set_labels("t10k-labels.idx1-ubyte");*/

	//network_model model = network_model::load_model_from_file("D:\\Users\\Matthew\\Neural Networks\\Models\\EMNIST Models", "emnist_cnn_mdl_3");
	network_model model;
	model.entry(train.load_size());

	tensor f0, f_b0, f1, f_b1, w0, b0, w1, b1;

	f0 = tensor::random_normal({ 5, 5, 3, 32 }, 0, 5e-2);
	f_b0 = tensor::zeros({ 32, 32, 32 });

	f1 = tensor::random_normal({ 5, 5, 32, 32 }, 0, 5e-2);
	f_b1 = tensor::zeros({ 16, 16, 32 });

	w0 = tensor::random_normal({ 384, 2048 }, 0, 5e-2);
	b0 = tensor::random_normal(384, 0, 5e-2);

	w1 = tensor::random_normal({ 10, 384 }, 0, 5e-2);
	b1 = tensor::random_normal(10, 0, 5e-2);/**/

	/*tensor w0, b0, w1, b1;

	w0 = tensor::random_normal({ 512, 784 }, 0, 5e-2);
	b0 = tensor::random_normal(512, 0, 5e-2);

	w1 = tensor::random_normal({ 10, 512 }, 0, 5e-2);
	b1 = tensor::random_normal(10, 0, 5e-2);*/

	/*std::ofstream out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_dnn_model/w0.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(w0.get_data())), sizeof(float) * w0.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_dnn_model/b0.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(b0.get_data())), sizeof(float) * b0.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_dnn_model/w1.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(w1.get_data())), sizeof(float) * w1.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_dnn_model/b1.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(b1.get_data())), sizeof(float) * b1.get_size());
	out_str.close();*/

	/*std::ofstream out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_cnn_model/f0.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(f0.get_data())), sizeof(float) * f0.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_cnn_model/f1.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(f1.get_data())), sizeof(float) * f1.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_cnn_model/w0.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(w0.get_data())), sizeof(float) * w0.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_cnn_model/b0.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(b0.get_data())), sizeof(float) * b0.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_cnn_model/w1.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(w1.get_data())), sizeof(float) * w1.get_size());
	out_str.close();

	out_str = std::ofstream("D:/Users/Matthew/Neural Networks/Testing Data/test_cnn_model/b1.bin", ios::binary);
	out_str.write(reinterpret_cast<char*>(reinterpret_cast<void*>(b1.get_data())), sizeof(float) * b1.get_size());
	out_str.close();*/

	model.conv2d(f0, f_b0, shape(2, 2));
	model.relu();
	model.max_pool(shape(3, 3), shape(2, 2));

	model.conv2d(f1, f_b1, shape(2, 2));
	model.relu();
	model.max_pool(shape(3, 3), shape(2, 2));

	model.flatten();

	model.matmul(w0);
	model.add(b0);
	model.relu();

	model.matmul(w1);
	model.add(b1);/**/
	
	/*model.conv2d(shape(5, 5), 32, SAME, variable_initialiser(0, 5e-2));
	model.relu();
	model.max_pool(shape(3, 3), shape(2, 2));

	model.conv2d(shape(5, 5), 32, SAME, variable_initialiser(0, 5e-2));
	model.relu();
	model.max_pool(shape(3, 3), shape(2, 2));

	model.flatten();
	model.dense(384);
	model.relu();
	model.dense(10);*/

	/*model.conv2d(shape(5, 5), 32, SAME, variable_initialiser(0, 5e-2));
	model.relu();
	model.max_pool(shape(2, 2), shape(2, 2));

	model.conv2d(shape(5, 5), 32, SAME, variable_initialiser(0, 5e-2));
	model.relu();
	model.max_pool(shape(2, 2), shape(2, 2));

	model.flatten();

	model.dense(384);
	model.relu();

	model.dense(10);*/

	model.set_output_function<softmax>();
	model.set_cost_function<softmax_cross_entropy>();
	model.set_optimiser(new adam(0.001));/**/

	model.initialise_model(256);

	analytics logger = analytics(PER_EPOCH, HIGH);
	logger.plot();

	model.add_logger(logger);

	model.train(train, 1);/**/

	float acc = model.evaluate(test) * 100;
	printf("Accuracy: %f\n", acc);

	model.uninitialise_model();
	train.close();
	test.close();

	return 0;
}