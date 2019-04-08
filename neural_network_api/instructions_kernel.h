#pragma once

#include <conio.h>
#include <algorithm>
#include <thrust\uninitialized_fill.h>
#include <stdio.h>
#include <stdexcept>
#include <time.h>
#include <curand.h>
#include <vector>

#include "kernel.h"

#include "timer.h"

using namespace std;

//define a new datatype to represent a byte (which is the same as a single unsigned char)

//Byte
//1 unsigned byte in memory
typedef unsigned char byte;

//device functions

//API FUNCTION
//Reduce the last warp to a single value with parallel reduction
template <unsigned int block_size>
__device__ void warp_reduce(volatile float * s_cost, int tid);

//API FUNCTION
//Reduce the last warp to a single value with parallel reduction,
//where an offset array is provided
template <unsigned int block_size>
__device__ void warp_reduce_to_zero(volatile float * s_cost);

//API FUNCTION
//Allocate a float pointer with the given size
void allocate_device_float_pointer(float ** d_pointer, size_t size);

//API FUNCTION
//Deallocate a float pointer
void deallocate_device_float_pointer(float * d_pointer);

//API FUNCTION
//Copies a host array into the device array with a specified pointer
void load_data_into_device(float * input_data, float * d_data_p, size_t size);

//API FUNCTION
//Copies a device array into the host array with a specified pointer
void retrieve_output_data(float * output_data, float * d_data_p, size_t size);

//API FUNCTION
//Copies a device array into another device array with a specified pointer
void copy_into_device_array(float * input_data, float * d_data_p, size_t size, size_t offset);

//API FUNCTION
//Gets a pseudorandom number generator with the specified seed
void get_prng(curandGenerator_t * prng, int seed);

//API FUNCTION
//Gets an array of scaled and offset random numbers
void random_host_array(curandGenerator_t prng, float * array_p, float scale, float offset, size_t size);

//API FUNCTION
//Gets an array of normally distributed random numbers
void random_normal_array(curandGenerator_t prng, float * array_p, float mean, float stddev, size_t size);

//API FUNCTION
//Fills an array on the device with the specfied value
void fill_device_array(float * d_array_p, float value, size_t size);

//API FUNCTION
//Adds a bias vector to each of the inputs
void add_matrices(float * d_input_p, float * d_out, float * d_bias_p, int size, int num);

//API FUNCTION
//Applies the ReLU activation function over each element of the input
void apply_relu(float * d_input_p, float * d_output_p, int size, float alpha);

//API FUNCTION
//Applies the Tanh activation function over each element of the input
void apply_tanh(float * d_input_p, float * d_output_p, int size);

//API FUNCTION
//Applies the Sigmoid activation function over each element of the input
void apply_sigmoid(float * d_input_p, float * d_output_p, int size);

//API FUNCTION
//Applies the Softmax function over the input
void apply_softmax(float * d_input_p, float * d_output_p, int input_size, int num, float beta);

//API FUNCTION
//Applies the ReLU derivative function over each element of the input
void relu_derivative(float * d_input_p, float * d_output_p, int size, float alpha);

//API FUNCTION
//Applies the Tanh derivative function over each element of the input
void tanh_derivative(float * d_input_p, float * d_output_p, int size);

//API FUNCTION
//Applies the Sigmoid derivative function over each element of the input
void sigmoid_derivative(float * d_input_p, float * d_output_p, int size);

//NOT IMPLEMENTED
//API FUNCTION
//Normalises the batch to centralise the mean and bring the standard
//deviation to 1
void batch_norm(float * d_input_p, float * d_output_p, int size, int num);

//NOT IMPLEMENTED
//API FUNCTION
//Calculates the derivative of the batch normalisation pass
void batch_norm_derivative(float * d_input_p, float * d_output_p, int size, int num);

//API FUNCTION
//Applies the Hadamard product to the two arrays, which performs elementwise multiplication
void hadamard_product(float * d_a, float * d_b, float * d_output_p, int in_size);

//API FUNCTION
//Transposes a matrix
void transpose(float * d_matrix_p, float * d_output_p, int rows, int cols);

//API FUNCTION
//Calculates the average vector from a matrix
void average_vector(float * d_matrix, float * d_output_p, int size, int num, int divisor);

//API FUNCTION
//Multiplies a float matrix by a scalar
void scalar_matrix_multiply_f(float * d_matrix, float * d_output_p, float scalar, int size);

//API FUNCTION
//Multiplies a byte matrix by a scalar
void scalar_matrix_multiply_b(byte * d_matrix, float * d_output_p, float scalar, int size);

//API FUNCTION
//Calculates the true mean of a vector
void average_value(float * d_input_p, float * average, int size);

//API FUNCTION
//Calculates the partial mean of a vector by specifying a different divisor
//from the size
void average_value(float * d_input_p, float * average, int size, float divisor);

//API FUNCTION
//Calculates the mean squared error cost between the two distributions
void squared_error_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size);

//API FUNCTION
//Calculates the derivative of the mean squared error between the distributions
void squared_error_cost_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size);

//API FUNCTION
//Calculates the cross entropy cost between the two distributions
void softmax_cross_entropy_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num);

//API FUNCTION
//Calculates the derivative of the cross entropy between the distributions
void softmax_cross_entropy_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num);

//API FUNCTION
//Returns an array of the indices of the maximum values for each input in the input array
void argmax(float * d_input, int * d_output, size_t input_size, size_t num);

//API FUNCTION
//Returns an array of the indices of the maximum values for each input in the input array, as well as
//another array of the maximum values themselvs
void argmax(float * d_input, float * d_max_vals, int * d_output, size_t input_size, size_t num);

//API FUNCTION
//Get the total number of equal elements in the arrays A and B
void comp_eq(int * a, int * b, unsigned int * res, size_t size);

//API FUNCTION
//Get the total number of equal elements in the arrays A and B
void comp_eq(int * a, float * b, unsigned int * res, size_t size);