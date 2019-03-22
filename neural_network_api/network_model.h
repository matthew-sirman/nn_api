#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <Windows.h>
#include <codecvt>
#include <time.h>
//#include "tensor.h"
#include "cost_functions.h"
//#include "instruction_functions.h"
#include "optimisers.h"
//#include "linear_algebra_ops.h"
#include "analytics.h"
#include "batch_iterator.h"

#include "timer.h"

using namespace nn;
using namespace std;

namespace nn {
	class NN_LIB_API network_model
	{
	public:
		network_model();
		~network_model();

		void add(tensor biases);
		void mul(tensor weights);
		//void dense(size_t in_size, size_t out_size);
		void dense(size_t input_size, size_t units);
		void conv2d(shape filter_shape, size_t n_filters, shape padding = shape(0, 0));
		void conv2d(tensor filter, shape padding = shape(0, 0));
		void pool(shape pool_size, shape stride);
		void flatten();
		void reshape(shape output_shape);
		void relu();
		void leaky_relu(float alpha);
		/*void softmax();
		void softmax(float beta);*/

		void function(instruction_function *func);

		template <typename T>
		inline void set_cost_function() { cost_func = new T(); }

		template <typename T>
		inline void set_output_function() { output_func = new T(); }

		inline void set_optimiser(optimiser * opt) { this->opt = opt; }

		void add_logger(analytics logger);

		void initialise_model();
		void initialise_model(size_t batch_size = 128);
		void initialise_model(shape input_shape, size_t batch_size = 128);
		void uninitialise_model();

		tensor run(tensor input);

		void train(tensor train_x, tensor train_y, int epochs);
		void train(batch_iterator &b_iter, int epochs);

		float get_accuracy(batch_iterator &b_iter);

		void write_model_to_file(string model_folder, string model_name);
		static network_model load_model_from_file(string model_folder, string model_name);

		trainable_function * TEST_get_t_func(int index) { return train_funcs[index]; }

	private:
		void calc_batch_gradient(float * d_x_batch, float * d_y_batch, size_t current_batch_size);

		vector<instruction_function*> instructions;
		vector<trainable_function*> train_funcs;
		vector<shape> layer_shapes;

		cost_function *cost_func = nullptr;
		output_function *output_func = nullptr;
		optimiser * opt;
		analytics *analytics_logger = nullptr;

		shape output_shape;

		size_t batch_size;

		size_t largest_layer_size;

		bool model_initialised = false;
	};
}

