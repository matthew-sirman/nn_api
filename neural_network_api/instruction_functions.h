#pragma once

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

	//API ENUMERATION
	//Instruction Function Type
	//Flags if a function is trainable or not
	enum instruction_function_type {
		TRAINABLE = 0x01
	};

	//API ENUMERATION
	//Function ID
	//Identifier for each of the network function types
	//availabe for stream serialisation
	enum function_id {
		ADD,
		MATMUL,
		RELU,
		L_RELU,
		BATCH_NORM,
		CONV_2D,
		POOL,
		RESHAPE,
		FLATTEN,
		TANH,
		SIGMOID
	};

	//API ENUMERATION
	//Output Function ID
	//Identifier for available output functions for
	//stream serialisation
	enum out_function_id {
		SOFTMAX = 0x00
	};

	//Padding Type
	//Padding types for convolutional layers
	//SAME-> Output image will have the same dimensions as the input image
	//VALID-> Output image will have dimensions of input_shape - filter_shape + 1
	enum padding_type {
		VALID,
		SAME
	};
	
	//Abstract base class for a serialisable function
	//Any derived class will have a method for serialising and deserialising
	//information to a byte stream buffer so that a model can be written to a
	//file.
	class serialisable_function
	{
	public:
		//API FUNCTION
		//Abstract Run method
		//Called to propagate through a network
		//Every network function should be able to be run
		virtual void run(float* input, size_t batch_size) = 0;

		//API FUNCTION
		//Get Serialise Size
		//Virtual function to get the size the function
		//will serialise to in the buffer, so a buffer can
		//be dynamically activated with a fixed size
		virtual size_t get_serialise_size();

		//API FUNCTION
		//Serialise
		//Serialise this function a byte stream
		virtual void serialise(char * stream_buff, size_t offset);

		//API FUNCTION
		//Deserialise
		//Set up this function from a byte stream
		virtual void deserialise(char * stream_buff, size_t offset);

		//Input shape - the shape this layer receives when run
		shape input_shape;

		//Output shape - the shape this layer returns when run
		shape output_shape;
	};

	//Abstract base class for any propagating network instruction function
	//Any derived class will be able to be added to a network model and
	//will be able to be back propagated
	class instruction_function : public serialisable_function
	{
	public:
		//Default Constructor
		instruction_function() {}
		//Destructor
		~instruction_function();

		//API FUNCTION
		//Abstract Run Derivative method
		//Called during the forward propagation stage but only during training
		//Sets up any information needed for the back propagation, such as caching
		//relevant information
		virtual void run_derivative(float* input) = 0;

		//API FUNCTION
		//Abstract Back Propagate method
		//Called during the backward propagation stage of training to compute
		//the partial derivatives with respect to the input it was given during
		//the forward pass.
		virtual void back_propagate(float * current_pds, int num) = 0;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		virtual void initialise(size_t batch_size);

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		virtual void uninitialise();

		//API FUNCTION
		//Set Input Shape
		//Sets the input shape and by default sets the output shape to the same as the input
		virtual inline void set_input_shape(shape input_shape) { this->input_shape = input_shape; this->output_shape = input_shape; }

		//API FUNCTION
		//Get Serialise Size
		//Adds on additional components to the serialise size after the size of the
		//base serialisable function and returns the new serialise size
		virtual size_t get_serialise_size() override;

		//API FUNCTION
		//Deserialise
		//Set up this function from a byte stream
		virtual void deserialise(char * stream_buffer, size_t offset) override;

		//API FUNCTION
		//Get Out Vector
		//Returns the computed cached vector after propagating
		float * get_out_vector();

		//API FUNCTION
		//Get Derivative Vector
		//Returns the computed derivative vector after propagating
		float * get_derivative_vector();

		//API FUNCTION
		//Get Type
		//Returns the type of this function (essentially if it is trainable)
		const inline int get_type() { return type; }

	protected:
		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream, but also takes an additional
		//function id parameter to specify the specific function which is being streamed
		void __serialise(char * stream_buffer, size_t offset, function_id func_id);

		//Out Vector
		//The output vector for this specific function declared on the device
		float *d_out_vector;

		//Derivative Vector
		//The derivative vector for this specific function declared on the device
		//(this vector is not necessarily used but is available if needed)
		float *d_der_vector;

		//Initialised
		//Flag to indicate if the function has been initialised
		//Defaults to false
		bool initialised = false;

		//Type
		//The type of function (trainable or non trainable)
		int type = 0;

		//Batch Size
		//The preset batch size with which the vectors are initialised
		//Sometimes the batch size for a specific iteration may be smaller
		//however, but it should never be larger
		size_t batch_size;
	};

	//Abstract base class for any network output function
	//Any derived class will be able to be added to the end of 
	//a network model and will be able to be called when the network
	//is run for predictions
	class output_function : public serialisable_function
	{
	public:
		//Default Constructor
		output_function() {}
		
		//Destructor
		~output_function() {}

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		virtual void initialise(shape input_shape, size_t batch_size);

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		virtual void uninitialise();

		//API FUNCTION
		//Get Out Vector
		//Returns the computed output vector after evaluating
		float * get_out_vector();

	protected:
		//Out Vector
		//The output vector for this specific output function declared on the device
		float * d_out_vector;

		//Initialised
		//Flag to indicate if the function has been initialised
		//Defaults to false
		bool initialised = false;

		//Batch Size
		//The preset batch size with which the vectors are initialised
		//Sometimes the batch size for a specific iteration may be smaller
		//however, but it should never be larger
		size_t batch_size;
	};

	//Abstract base class for any trainable network instruction function
	//Any derived class will be able to be added to a network model and
	//will be able to be back propagated, as well as have some weight(s)
	//which will be updated and trained
	class trainable_function : public instruction_function {
	public:
		//Default Constructor
		trainable_function() { type |= instruction_function_type::TRAINABLE; }

		//Constructor given initial training tensor
		//The provided tensor will be automatically updated during training.
		//To prevent the tensor being updated, the train function can be locked
		//with the lock() method
		trainable_function(tensor t);

		//Destructor
		~trainable_function() {};

		//API FUNCTION
		//Abstract Run Train Derivative method
		//Called during the forward propagation stage but only during training
		//Sets up any information needed for the training in back propagation, 
		//such as caching relevant information
		virtual void run_train_derivative(float* input, int batch_size) = 0;

		//API FUNCTION
		//Get Train Vector
		//Returns the training vector. This is the vector associated with the training
		//tensor, but as a memory device vector rather than a tensor construct
		inline float * get_train_vector() { return train_tensor.get_dev_pointer(); }

		//API FUNCTION
		//Get Derivative Vector
		//Returns the vector of all the calculated derivatives with respect to the training tensor.
		//This is the vector associated with the derivatives tensor, but as a memory device vector
		//rather than a tensor construct
		inline float * get_train_derivative_vector() { return derivatives.get_dev_pointer(); }

		//API FUNCTION
		//Get Train Tensor
		//Returns the train tensor as a tensor
		inline tensor get_train_tensor() { return train_tensor; }

		//API FUNCTION
		//Get Train Tensor Size
		//Wrapping helper function which simply returns the train tensor's get_size() method
		inline size_t get_train_tensor_size() { return train_tensor.get_size(); }

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Get Serialise Size
		//Adds on additional components to the serialise size after the size of the
		//base serialisable function and returns the new serialise size
		virtual size_t get_serialise_size() override;

		//API FUNCTION
		//Deserialise
		//Set up this function from a byte stream
		virtual void deserialise(char * stream_buffer, size_t offset) override;

		//API FUNCTION
		//Average Partial Derivatives
		//Called when the back propagation reaches this function. Calculates
		//the final partial derivatives with respect to the train tensor and caches
		//them
		virtual void avg_partial_derivatives(float * current_pds, int num) = 0;

		//Lock
		//Locks the training tensor, so it is unaltered during the training
		inline void lock() { is_locked = true; }

		//Unlock
		//Unlocks the training tensor, so it is updated during the training
		inline void unlock() { is_locked = false; }

		//Locked
		//Returns whether the training tensor is locked or not
		inline bool locked() { return is_locked; }

	protected:
		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream, but also takes an additional
		//function id parameter to specify the specific function which is being streamed
		void __serialise(char * stream_buffer, size_t offset, function_id func_id);

		//Train Tensor
		//The tensor which holds the weight(s) for this model which are updated
		//during the training phase, provided this function is not locked
		tensor train_tensor;

		//Derivatives
		//The tensor which holds the derivatives with respect to the train tensor
		//for use in the optimiser function
		tensor derivatives;

		//Partial Derivative Vector
		//The partial derivative vector for this specific function declared on the device
		//(this vector is not necessarily used but is available if needed)
		float * d_pder_vector;
	private:
		//is the train tensor locked or not?
		bool is_locked = false;
	};

	//API CLASS
	//Add Function
	//Trainable layer function which adds an input vector to a trainable tensor
	//of biases
	class add_function : public trainable_function {
	public:
		//Default Constructor
		add_function() {};

		//Constructor specifying the shape of the bias tensor
		add_function(shape bias_size);

		//Constructor specifying the actual bias tensor to use
		add_function(tensor biases);

		//Destructor
		~add_function();

		//API FUNCTION
		//Run
		//Add the biases to each input in the batch
		void run(float* input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation.
		//As the derivatives for addition are always 1, no operation is required
		void run_derivative(float* input) override {};

		//API FUNCTION
		//Run Train Derivative
		//Calculate relevant information for backpropagation training.
		//As the derivatives for addition are always 1, no operation is required
		void run_train_derivative(float* input, int batch_size) override {};

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		//As the derivatives for addition are always 1, no operation is required
		void back_propagate(float * current_pds, int num) override {};

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Average Partial Derivatives
		//Called when the back propagation reaches this function. Calculates
		//the final partial derivatives with respect to the bias train tensor 
		//and caches them
		void avg_partial_derivatives(float * current_pds, int num) override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;
	private:
		//vector of temporary derivates needed before calculating the average
		float * d_derivatives;
	};

	//API CLASS
	//Matmul Function
	//Trainable layer function which performs matrix multiplication between an input
	//vector and a trainable tensor of weights
	class matmul_function : public trainable_function {
	public:
		//Default constructor
		matmul_function() {};

		//Constructor specifying the size of the weight matrix
		matmul_function(size_t weight_rows, size_t weight_cols);

		//Constructor specifying the actual weight matrix to use
		matmul_function(tensor weights);

		//Destructor
		~matmul_function();

		//API FUNCTION
		//Run
		//Multiplies each input in the batch by the weight matrix
		void run(float* input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float* input) override {};

		//API FUNCTION
		//Run Train Derivative
		//Calculate relevant information for backpropagation training
		void run_train_derivative(float* input, int batch_size) override;

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int num) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Set Input Shape
		//Overrides the set input shape method such that it doesn't set the output shape to be the same.
		//The output shape for the matmul function will mostly be different from its input
		inline void set_input_shape(shape input_shape) override { this->input_shape = input_shape; }

		//API FUNCTION
		//Average Partial Derivatives
		//Called when the back propagation reaches this function. Calculates
		//the final partial derivatives with respect to the weight train tensor 
		//and caches them
		void avg_partial_derivatives(float * current_pds, int num) override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;
	private:
		//device matrix preset to the training matrix 
		d_matrix<float> d_mat;

		//device matrix for the output of the forward propagation
		d_matrix<float> d_out_vec;

		//temporary variable to hold the back propagation result before writing back
		//to the true partial derivative vector to avoid read/write clashes
		float * d_bp_temp;
	};

	//API CLASS
	//Conv2D Function
	//Trainable layer function which performs 2D convolution over the image space
	//with a tensor of trainable filters
	class conv2d_function : public trainable_function 
	{
	public:
		//Default Constructor
		conv2d_function() {};

		//Constructor specifying the input shape, filter shape, number of filters and padding size
		conv2d_function(shape input_shape, shape filter_shape, size_t n_filters = 1, shape padding = (0, 0));

		//Constructor specifying the actual filter tensor to use and the padding size
		conv2d_function(tensor filter, shape padding = (0, 0));

		//Destructor
		~conv2d_function();

		//API FUNCTION
		//Run
		//Convolves each filter over the input images and writes each result to the output
		void run(float * input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float * input) override {};

		//API FUNCTION
		//Run Train Derivative
		//Calculate relevant information for backpropagation training
		void run_train_derivative(float * input, int num) override;

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int num) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Average Partial Derivatives
		//Called when the back propagation reaches this function. Calculates
		//the final partial derivatives with respect to the filter train tensor 
		//and caches them
		void avg_partial_derivatives(float * current_pds, int num) override;

		//API FUNCTION
		//Get Serialise Size
		//Adds on additional components to the serialise size after the size of the
		//base serialisable function and returns the new serialise size
		size_t get_serialise_size() override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;

		//API FUNCTION
		//Deserialise
		//Deseirialses an instruction function from a byte stream
		void deserialise(char * stream_buffer, size_t offset) override;

		//API FUNCTION
		//Set Input Shape
		//Overrides the set input shape method such that it doesn't set the output shape to be the same
		void set_input_shape(shape input_shape) override;

		//API FUNCTION
		//Get Filter
		//Returns the train tensor, which is the filter
		inline tensor & get_filter() { return train_tensor; }
	private:
		//the shape of the filter (width, height, depth)
		shape filter_shape;

		//the padding for the filter
		shape padding;

		//temporary variable to hold the back propagation result before writing back
		//to the true partial derivative vector to avoid read/write clashes
		float * d_tmp_backprop_output;
	};

	//API CLASS
	//Max Pool Function
	//Layer function which returns an output map with the maximum value of each
	//pool sized pool from an input map
	class max_pool_function : public instruction_function
	{
	public:
		//Default Constructor
		max_pool_function() {};

		//Constructor specifying pool size and stride
		max_pool_function(shape pool_size, shape stride);

		//Destructor
		~max_pool_function();

		//API FUNCTION
		//Run
		//Performs max pooling convolution over each input
		void run(float* input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float* input) override {};

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int num) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Set Input Shape
		//Overrides the set input shape method such that it doesn't set the output shape to be the same
		void set_input_shape(shape input_shape) override;

		//API FUNCTION
		//Get Serialise Size
		//Adds on additional components to the serialise size after the size of the
		//base serialisable function and returns the new serialise size
		size_t get_serialise_size() override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;

		//API FUNCTION
		//Deserialise
		//Deseirialses an instruction function from a byte stream
		void deserialise(char * stream_buffer, size_t offset) override;
	private:
		//the size of each pool
		shape pool_size;

		//the stride which the pool steps over
		shape stride;

		//padding for the pool if it goes over the edge
		shape padding;

		//mask pointer for back propagation
		int * d_mask;
	};

	//API CLASS
	//Reshape Function
	//Layer function which changes the layer shape from one shape to another
	//shape with equal total size (width * height * depth)
	class reshape_function : public instruction_function
	{
	public:
		//Default Constructor
		reshape_function() {};

		//Constructor specifying the input shape and output shape for the reshape function
		reshape_function(shape in_shape, shape out_shape) { this->input_shape = in_shape; this->output_shape = out_shape; };

		//Destructor
		~reshape_function() {};

		//API FUNCTION
		//Run
		//Performs max pooling convolution over each input
		inline void run(float* input, size_t batch_size) override {
			cuda_safe_call(cudaMemcpy(d_out_vector, input, sizeof(float) * input_shape.size() * batch_size, cudaMemcpyDeviceToDevice));
		}

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		//Nothing required for this function (as it is entirely linear)
		inline void run_derivative(float * input) override {};

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		//Nothing required for this function (as it is entirely linear)
		inline void back_propagate(float * current_pds, int num) override {};

		//API FUNCTION
		//Set Input Shape
		//Overrides the set input shape method such that it doesn't set the output shape to be the same
		virtual inline void set_input_shape(shape input_shape) override { this->input_shape = input_shape; }

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		inline void serialise(char * stream_buffer, size_t offset) override { __serialise(stream_buffer, offset, function_id::RESHAPE); }
	};

	//API CLASS
	//Flatten Function
	//Layer function which reshapes a layer such that it is one dimensional but
	//has the same total size
	class flatten_function : public reshape_function
	{
	public:
		//Default Constructor
		flatten_function() {};

		//Constructor specifying only input shape as output shape is determined only by input size
		flatten_function(shape in_shape) : reshape_function(in_shape, shape(in_shape.size())) {};

		//Destructor
		~flatten_function() {};

		//API FUNCTION
		//Set Input Shape
		//Overrides the set input shape method such that it sets the output shape correctly
		inline void set_input_shape(shape input_shape) override { this->input_shape = input_shape; this->output_shape = shape(input_shape.size()); }

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		inline void serialise(char * stream_buffer, size_t offset) override { __serialise(stream_buffer, offset, function_id::FLATTEN); }
	};

	//API CLASS
	//ReLU Function
	//Layer function which maps the ReLU activation over the input space, where
	//the function max(0, x) is applied elementwise to the input
	class relu_function : public instruction_function {
	public:
		//Default Constructor
		relu_function() : relu_function(1) {}

		//Constructor specifying the input shape
		relu_function(shape input_shape);

		//Destructor
		~relu_function();

		//API FUNCTION
		//Run
		//Performs elementwise ReLU over input
		void run(float* input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float* input) override;

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int num) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;
	};

	//API CLASS
	//Leaky ReLU Function
	//Layer function which maps the Leaky ReLU activation over the input space, where
	//the function max(alpha, x) is applied elementwise to the input, where alpha is a
	//non-zero constant
	class leaky_relu_function : public instruction_function {
	public:
		//Default Constructor
		leaky_relu_function() {};

		//Constructor specifying the alpa constant
		leaky_relu_function(float alpha) : leaky_relu_function(1, alpha) {};

		//Constructor specifying the input shape and alpha constant
		leaky_relu_function(shape input_shape, float alpha);

		//Destructor
		~leaky_relu_function();

		//API FUNCTION
		//Run
		//Performs elementwise Leaky ReLU over input
		void run(float* input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float* input) override;

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int num) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//Alpha
		//Constant alpha for the "leak" of the function
		float alpha;

		//API FUNCTION
		//Get Serialise Size
		//Adds on the alpha value size to the serialise size after the size of the
		//base serialisable function and returns the new serialise size
		size_t get_serialise_size() override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;

		//API FUNCTION
		//Deserialise
		//Deseirialses an instruction function from a byte stream
		void deserialise(char * stream_buffer, size_t offset) override;
	};

	//API CLASS
	//Tanh Function
	//Layer function which maps the Tanh activation over the input space, where
	//the function Tanh(x) (hyperbolic tangent) is applied elementwise to the input
	class tanh_function : public instruction_function {
	public:
		//Default Constructor
		tanh_function() : tanh_function(1) {};

		//Constructor specifying the input shape
		tanh_function(shape input_shape);

		//Destructor
		~tanh_function();

		//API FUNCTION
		//Run
		//Performs elementwise Tanh over input
		void run(float * input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float * input) override;

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int batch_size) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;
	};

	//API CLASS
	//Sigmoid Function
	//Layer function which maps the ReLU activation over the input space, where
	//the function 1/(1 + e^-x) is applied elementwise to the input
	class sigmoid_function : public instruction_function {
	public:
		//Default Constructor
		sigmoid_function() : sigmoid_function(1) {};

		//Constructor specifying the input shape
		sigmoid_function(shape input_shape);

		//Destructor
		~sigmoid_function();

		//API FUNCTION
		//Run
		//Performs elementwise Sigmoid over input
		void run(float * input, size_t batch_size) override;

		//API FUNCTION
		//Run Derivative
		//Calculate relevant information for backpropagation
		void run_derivative(float * input) override;

		//API FUNCTION
		//Back propagate
		//Calculate the partial derivative with respect to the input
		void back_propagate(float * current_pds, int batch_size) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//API FUNCTION
		//Serialise
		//Serialises an instruction function into a byte stream
		void serialise(char * stream_buffer, size_t offset) override;
	};

	//NOT IMPLEMENTED
	//API CLASS
	//LSTM Function
	//Trainable layer function which adds an LSTM (Long Short-Term Memory) cell
	//to the network model to allow for recurrent network learning
	class lstm_function : public trainable_function {
	public:
		lstm_function();
		~lstm_function();


	private:

	};

	//NOT IMPLEMENTED
	//API CLASS
	//Batch Normalisation Function
	//Layer function which normalises the input by transforming the mean to 0 and
	//standard deviation to 1
	class batch_normalisation_function : public instruction_function {
	public:
		batch_normalisation_function() : batch_normalisation_function(1) {};
		batch_normalisation_function(size_t input_size);

		void run(float * input, size_t batch_size) override;
		void run_derivative(float * input) override;

		void back_propagate(float * current_pds, int num) override;

		void initialise(size_t batch_size) override;
		void uninitialise() override;

		void serialise(char * stream_buffer, size_t offset) override;
	};

	//API CLASS
	//Softmax Function
	//Output function which applies the softmax function to each output vector in
	//a batch, where the function e^xi/Sum[e^x] is applied over each batch element
	//vector
	class softmax : public output_function {
	public:
		//Default Constructor
		softmax() : softmax(1) {};

		//Constructor specifying the input size
		softmax(size_t input_size);

		//Destructor
		~softmax();

		//API FUNCTION
		//Run
		//Performs softmax over the batch to each element
		void run(float* input, size_t batch_size) override;

		//API FUNCTION
		//Initialise
		//Initialises the function by declaring and allocating memory and initialising
		//the batch size for the function
		void initialise(shape input_shape, size_t batch_size) override;

		//API FUNCTION
		//Uninitialise
		//Uninitialises the function by dereferencing memory
		void uninitialise() override;

		//Function ID
		//Constant set to SOFTMAX to specify that this is a softmax function
		const int func_id = out_function_id::SOFTMAX;
	};

}

