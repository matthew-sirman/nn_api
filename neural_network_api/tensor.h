#pragma once

#include <curand_kernel.h>

#include <vector>
#include "instructions_kernel.h"

using namespace std;

//Constant variable to determine if a non-zero seed
//should be used for initialisation
//If false, the seed will be 0 so all the variables created
//"random" tensors will be the same from run to run
constexpr bool TRUE_RAND = true;

//Tensor
//Holds a multidimensional tensor of numbers with a given shape.
//Can contain any number of dimensions
class tensor
{
public:
	//Default constructor
	tensor() {};

	//Constructor specifying the size of a 1D tensor (vector)
	tensor(size_t size);

	//Constructor specifying the size of a 1D tensor (vector) and its starting data
	tensor(size_t size, float *data);

	//Constructor specifying the size of a multidimensional tensor
	tensor(vector<size_t> shape);

	//Constructor specifying the size of a multidimensional tensor and its starting data
	tensor(vector<size_t> shape, float *data);

	//Destructor
	~tensor() {};

	//Random
	//Static method to return a new 1D vector tensor with a set of
	//uniformly distributed random numbers between 0 and 1
	static tensor random(size_t size);

	//Random
	//Static method to return a new 1D vector tensor with a set of
	//uniformly distributed random numbers between min and max
	static tensor random(size_t size, float min, float max);

	//Random
	//Static method to return a new multidimensional tensor with a set of
	//uniformly distributed random numbers between 0 and 1
	static tensor random(vector<size_t> shape);

	//Random
	//Static method to return a new multidimensional tensor with a set of
	//uniformly distributed random numbers between min and max
	static tensor random(vector<size_t> shape, float min, float max);
	
	//Random Normal
	//Static method to return a new 1D vector tensor with a set of
	//normally distributed random numbers with the specified mean and standard
	//deviation
	static tensor random_normal(size_t size, float mean = 0.0f, float stddev = 1.0f);

	//Random Normal
	//Static method to return a new multidimensional tensor with a set of
	//normally distributed random numbers with the specified mean and standard
	//deviation
	static tensor random_normal(vector<size_t> shape, float mean = 0.0f, float stddev = 1.0f);
	
	//Zeros
	//Static method to return a new 1D vector tensor filled with 0s
	static tensor zeros(size_t size);

	//Value
	//Static method to return a new 1D vector tensor filled with a set value
	static tensor value(size_t size, float value);

	//Zeros
	//Static method to return a new multidimensional tensor filled with 0s
	static tensor zeros(vector<size_t> shape);

	//Zeros
	//Static method to return a new multidimensional tensor filled with a set value
	static tensor value(vector<size_t> shape, float value);

	//One Hot
	//Static method to return a new tensor with all 0s with 1 value as 1, at the index
	//specified by the "hot" parameter
	static tensor one_hot(size_t size, int hot);

	//Stack
	//Static method to return a new tensor which is a stack of all the supplied tensors.
	//Supplied tensors must all have the same shape
	static tensor stack(vector<tensor> tensors);

	//Reshape
	//Set the tensor to a new shape with the same size
	void reshape(vector<size_t> new_shape);

	//Set
	//Sets a value at a given index in a 1D vector tensor
	void set(int index, float value);

	//Set
	//Sets a value at a given index in a multidimensional tensor
	void set(vector<int> index, float value);

	//Get
	//Returns the value at a given index in a 1D vector tensor
	float get(int index);

	//Get
	//Returns the value at a given index in a multidimensional tensor
	float get(vector<int> index);

	//API FUNCTION
	//Serialise
	//Serialises the shape and values of this tensor into a byte stream
	void serialise(char * stream_buffer, size_t offset);

	//API FUNCTION
	//Deserialise
	//Deserialises the shape and values from a byte stream into this tensor
	void deserialise(char * stream_buffer, size_t offset);

	//Get Transpose
	//Returns the transpose of this 2D matrix tensor
	tensor get_transpose();

	//Initialise
	//Initialises the tensor for use on the device
	void initialise();

	//Uninitialise
	//Uninitliases the tensor by dereferncing memory
	void uninitialise();

	//API FUNCTION
	//Get Data
	//Returns the raw unformatted host data in this tensor as a float
	//array
	float *get_data();

	//API FUNCTION
	//Get Device Pointer
	//Returns the raw unformatted device data in this tensor as a float
	//array
	float *get_dev_pointer();

	//API FUNCTION
	//Set Device Pointer
	//Sets the raw unformatted device data to the tensor passed in
	void set_dev_pointer(float * d_data_p);

	//Get Dimensions
	//Returns the number of dimensions this tensor has
	size_t get_dimensions();

	//Get Shape
	//Returns the shape of this tensor
	vector<size_t> get_shape();

	//Get Size
	//Gets the total size of this tensor, which is the product of each 
	//dimension of its shape
	size_t get_size();

	//API FUNCTION
	//Set Shape
	//VOLATILE function to overwrite the shape of this tensor without checking
	//that the size is the same before and after.
	//For safe reshaping, use the reshape() method
	void set_shape(vector<size_t> shape);

private:
	//returns the linear index for a given multidimensional index
	int get_linear_offset(vector<int> index);

	//pointer to the actual raw linear data for this tensor
	float *data;

	//pointer to the actual raw linear data on the device for this tensor
	float *d_data;

	//vector of sizes representing the shape of this tensor
	vector<size_t> shape;

	//flag to indicate if the tensor is initialised for device usage
	bool initialised = false;

	//static flag to determine if the PRNG has been initialsed by any tensor
	static bool random_initialised;

	//static pseudorandom number generator for use across all tensors
	static curandGenerator_t prng;
};

