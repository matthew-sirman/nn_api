#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <stdexcept>
#include <string>
#include <iostream>
#include "instructions_kernel.h"
#include "linear_algebra_ops.h"
#include "conv_ops_2d.h"
#include "tensor.h"
#include "shape.h"

using namespace std;

namespace nn {

	enum NN_LIB_API instruction_function_type {
		TRAINABLE = 0x01
	};

	enum NN_LIB_API function_id {
		ADD = 0x00,
		MUL = 0x01,
		RELU = 0x02,
		L_RELU = 0x03,
		BATCH_NORM = 0x04,
		CONV_2D = 0x05,
		POOL = 0x06,
		RESHAPE = 0x07,
		FLATTEN = 0x08
	};

	enum NN_LIB_API out_function_id {
		SOFTMAX = 0x00
	};
	
	class NN_LIB_API serialisable_function
	{
	public:
		virtual void run(float* input) = 0;
		virtual void run(float* input, int batch_size) = 0;

		virtual size_t get_serialise_size();
		virtual void serialise(char * stream_buff, size_t offset);
		virtual void deserialise(char * stream_buff, size_t offset);

		shape input_shape, output_shape;
	};

	class NN_LIB_API instruction_function : public serialisable_function
	{
	public:
		instruction_function() {}
		~instruction_function();

		virtual void run_derivative(float* input) = 0;

		virtual void back_propagate(float * current_pds, int num) = 0;

		virtual void initialise();
		virtual void initialise(size_t batch_size);
		virtual void uninitialise();

		virtual inline void set_input_shape(shape input_shape) { this->input_shape = input_shape; this->output_shape = input_shape; }

		virtual size_t get_serialise_size() override;
		virtual void deserialise(char * stream_buffer, size_t offset) override;

		float * get_out_vector();
		float * get_derivative_vector();

		//size_t input_size, output_size;

		const inline int get_type() { return type; }

	protected:
		void __serialise(char * stream_buffer, size_t offset, function_id func_id);

		float *d_out_vector;
		float *d_der_vector;
		bool initialised = false;

		int type = 0;

		size_t batch_size;
	};

	class NN_LIB_API output_function : public serialisable_function
	{
	public:
		output_function() {}
		~output_function() {}

		//virtual void run(float * input) = 0;
		//virtual void run(float * input, int batch_size) = 0;

		virtual void initialise();
		virtual void initialise(shape input_shape, size_t batch_size);
		virtual void uninitialise();

		float * get_out_vector();

	protected:
		float * d_out_vector;

		bool initialised = false;

		size_t batch_size;
	};

	class NN_LIB_API trainable_function : public instruction_function {
	public:
		trainable_function() { type |= instruction_function_type::TRAINABLE; }
		trainable_function(tensor t);
		~trainable_function() {};

		virtual void run_train_derivative(float* input, int batch_size) = 0;

		inline float * get_train_derivative_vector() { return d_pder_vector; }
		inline float * get_momentum() { return d_momentum; }
		inline float * get_velocity() { return d_velocity; }
		inline float * get_train_vector() { return train_tensor.get_dev_pointer(); }
		inline float * get_derivative_vector() { return derivatives.get_dev_pointer(); }

		inline tensor get_train_tensor() { return train_tensor; }
		inline size_t get_train_tensor_size() { return train_tensor.get_size(); }

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		virtual size_t get_serialise_size() override;
		virtual void deserialise(char * stream_buffer, size_t offset) override;

		virtual void avg_partial_derivatives(float * current_pds, int num) = 0;

		//virtual void train_function(float learning_rate, float momentum) = 0;

	protected:
		void __serialise(char * stream_buffer, size_t offset, function_id func_id);

		tensor train_tensor;
		tensor derivatives;
		float * d_pder_vector;

		float * d_derivatives;
		float * d_avg_derivatives;

		float * d_momentum;
		float * d_velocity;
	};

	class add_function : public trainable_function {
	public:
		add_function() {};
		add_function(size_t bias_size);
		add_function(tensor biases);
		~add_function();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		void run_derivative(float* input) override;
		void run_train_derivative(float* input, int batch_size) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		void avg_partial_derivatives(float * current_pds, int num) override;

		void serialise(char * stream_buffer, size_t offset) override;

		//void train_function(float learning_rate, float momentum) override;
	};

	class mul_function : public trainable_function {
	public:
		mul_function() {};
		mul_function(size_t weight_rows, size_t weight_cols);
		mul_function(tensor weights);
		~mul_function();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		void run_derivative(float* input) override;
		void run_train_derivative(float* input, int batch_size) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		inline void set_input_shape(shape input_shape) override { this->input_shape = input_shape; }

		void avg_partial_derivatives(float * current_pds, int num) override;

		void serialise(char * stream_buffer, size_t offset) override;

		//void train_function(float learning_rate, float momentum) override;
	private:
		//float * d_partial_mul_outputs;
		//float * d_transpose_matrix;
		//float * current_pds_t;
		d_matrix<float> d_mat;
		d_matrix<float> d_out_vec;

		float * d_bp_temp;
	};

	class conv2d_function : public trainable_function 
	{
	public:
		conv2d_function() {};
		conv2d_function(shape input_shape, shape filter_shape, size_t n_filters = 1, shape padding = (0, 0));
		conv2d_function(tensor filter, shape padding = (0, 0));
		~conv2d_function();

		void run(float * input) override;
		void run(float * input, int batch_size) override;
		void run_derivative(float * input) override;
		void run_train_derivative(float * input, int num) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		void avg_partial_derivatives(float * current_pds, int num) override;

		size_t get_serialise_size() override;
		void serialise(char * stream_buffer, size_t offset) override;
		void deserialise(char * stream_buffer, size_t offset) override;

		void set_input_shape(shape input_shape) override;

		inline tensor & get_filter() { return train_tensor; }
	private:
		shape filter_shape;
		shape padding;

		float * d_tmp_backprop_output;
	};

	class pool_function : public instruction_function
	{
	public:
		pool_function() {};
		pool_function(shape pool_size, shape stride);
		~pool_function();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		void run_derivative(float* input) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise();
		void initialise(size_t batch_size);
		void uninitialise();

		void set_input_shape(shape input_shape) override;
		
		size_t get_serialise_size() override;
		void serialise(char * stream_buffer, size_t offset) override;
		void deserialise(char * stream_buffer, size_t offset) override;
	private:
		shape pool_size;
		shape stride;

		int * d_mask;

		int * test;
	};

	class reshape_function : public instruction_function
	{
	public:
		reshape_function() {};
		reshape_function(shape in_shape, shape out_shape) { this->input_shape = in_shape; this->output_shape = out_shape; };
		~reshape_function() {};

		inline void run(float* input) override { run(input, 1); }
		inline void run(float* input, int batch_size) override { 
			cuda_safe_call(cudaMemcpy(d_out_vector, input, sizeof(float) * input_shape.size() * batch_size, cudaMemcpyDeviceToDevice)); 
		}
		inline void run_derivative(float * input) override {};
		inline void back_propagate(float * current_pds, int num) override {};
		virtual inline void set_input_shape(shape input_shape) override { this->input_shape = input_shape; }

		inline void serialise(char * stream_buffer, size_t offset) override { __serialise(stream_buffer, offset, function_id::RESHAPE); }
	};

	class flatten_function : public reshape_function
	{
	public:
		flatten_function() {};
		flatten_function(shape in_shape) : reshape_function(in_shape, shape(in_shape.size())) {};
		~flatten_function() {};

		inline void set_input_shape(shape input_shape) override { this->input_shape = input_shape; this->output_shape = shape(input_shape.size()); }

		inline void serialise(char * stream_buffer, size_t offset) override { __serialise(stream_buffer, offset, function_id::FLATTEN); }
	};

	class relu_function : public instruction_function {
	public:
		relu_function();
		relu_function(size_t input_size);
		~relu_function();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		void run_derivative(float* input) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		void serialise(char * stream_buffer, size_t offset) override;
	};

	class leaky_relu_function : public instruction_function {
	public:
		leaky_relu_function() {};
		leaky_relu_function(float alpha);
		leaky_relu_function(size_t input_size, float alpha);
		~leaky_relu_function();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		void run_derivative(float* input) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		float alpha;

		size_t get_serialise_size() override;
		void serialise(char * stream_buffer, size_t offset) override;
		void deserialise(char * stream_buffer, size_t offset) override;
	};

	class batch_normalisation_function : public instruction_function {
	public:
		batch_normalisation_function();
		batch_normalisation_function(size_t input_size);

		void run(float * input) override;
		void run(float * input, int batch_size) override;
		void run_derivative(float * input) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		void serialise(char * stream_buffer, size_t offset) override;
	};

	/*class softmax_function : public instruction_function {
	public:
		softmax_function();
		softmax_function(size_t input_size);
		softmax_function(size_t input_size, float beta);
		~softmax_function();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		void run_derivative(float* input) override;

		void initialise() override;
		void initialise(size_t batch_size) override;
		void uninitialise() override;

		float beta;

	private:
		float * d_partial_mul_outputs;
	};*/

	class NN_LIB_API softmax : public output_function {
	public:
		softmax();
		softmax(size_t input_size);
		~softmax();

		void run(float* input) override;
		void run(float* input, int batch_size) override;
		//void run_derivative(float* input) override;

		void initialise() override;
		void initialise(shape input_shape, size_t batch_size) override;
		void uninitialise() override;

		const int func_id = out_function_id::SOFTMAX;
	};

}

