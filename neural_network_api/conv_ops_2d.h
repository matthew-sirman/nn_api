#pragma once

#include "kernel.h"

#include "shape.h"

//this file contains definitions for API functions so the user should never call these functions directly.

//hyperparameters
//allow for fine tuning to improve efficiency

//hyperparameter constants for convolutional forward propagations
constexpr auto CONV_BLOCK_SIZE_N = 32;
constexpr auto CONV_BLOCK_SIZE_M = 16;
constexpr auto CONV_BLOCK_SIZE_K = 4;
constexpr auto CONV_THREAD_BLOCK_N = 4;
constexpr auto CONV_THREAD_BLOCK_M = 2;
constexpr auto CONV_THREAD_BLOCK_K = 4;
constexpr auto CONV_BATCH_DEPTH = 1;
constexpr auto CONV_THREAD_SIZE_N = CONV_BLOCK_SIZE_N / CONV_THREAD_BLOCK_N;
constexpr auto CONV_THREAD_SIZE_M = CONV_BLOCK_SIZE_M / CONV_THREAD_BLOCK_M;
constexpr auto CONV_THREAD_SIZE_K = CONV_BLOCK_SIZE_K / CONV_THREAD_BLOCK_K;

//hyperparameter constants for filters
constexpr auto FILTER_BLOCK_SIZE_N = 8;
constexpr auto FILTER_BLOCK_SIZE_M = 8;
constexpr auto FILTER_BLOCK_SIZE_K = CONV_BLOCK_SIZE_K;
constexpr auto FILTER_THREAD_BLOCK_N = FILTER_BLOCK_SIZE_N / CONV_THREAD_SIZE_N;
constexpr auto FILTER_THREAD_BLOCK_M = FILTER_BLOCK_SIZE_M / CONV_THREAD_SIZE_M;

//hyperparameter constants for outer convolutions
//used in back propagation for training purposes
constexpr auto CONV_OUTER_BLOCK_SIZE_N = 32;
constexpr auto CONV_OUTER_BLOCK_SIZE_M = 16;
constexpr auto CONV_OUTER_BLOCK_SIZE_K = 1;
constexpr auto CONV_OUTER_THREAD_BLOCK_N = 4;
constexpr auto CONV_OUTER_THREAD_BLOCK_M = 4;
constexpr auto CONV_OUTER_THREAD_BLOCK_K = 1;
constexpr auto CONV_OUTER_THREAD_SIZE_N = CONV_OUTER_BLOCK_SIZE_N / CONV_OUTER_THREAD_BLOCK_N;
constexpr auto CONV_OUTER_THREAD_SIZE_M = CONV_OUTER_BLOCK_SIZE_M / CONV_OUTER_THREAD_BLOCK_M;
constexpr auto CONV_OUTER_THREAD_SIZE_K = CONV_OUTER_BLOCK_SIZE_K / CONV_OUTER_THREAD_BLOCK_K;

//hyperparameter constants for derivative convolutions
//used in back propagation for training purposes
constexpr auto CONV_DER_BLOCK_SIZE_N = 32;
constexpr auto CONV_DER_BLOCK_SIZE_M = 32;
constexpr auto CONV_DER_BLOCK_SIZE_K = 4;
constexpr auto CONV_DER_THREAD_BLOCK_N = 4;
constexpr auto CONV_DER_THREAD_BLOCK_M = 4;
constexpr auto CONV_DER_THREAD_BLOCK_K = 2;
constexpr auto CONV_DER_THREAD_SIZE_N = CONV_DER_BLOCK_SIZE_N / CONV_DER_THREAD_BLOCK_N;
constexpr auto CONV_DER_THREAD_SIZE_M = CONV_DER_BLOCK_SIZE_M / CONV_DER_THREAD_BLOCK_M;
constexpr auto CONV_DER_THREAD_SIZE_K = CONV_DER_BLOCK_SIZE_K / CONV_DER_THREAD_BLOCK_K;

//hyperparameter constants for pooling layers
constexpr auto POOL_BLOCK_SIZE_N = 16;
constexpr auto POOL_BLOCK_SIZE_M = 16;
constexpr auto POOL_BLOCK_DEPTH = 16;
constexpr auto POOL_THREAD_BLOCK_N = 4;
constexpr auto POOL_THREAD_BLOCK_M = 4;
constexpr auto POOL_THREAD_BLOCK_K = 2;
constexpr auto POOL_THREAD_SIZE_N = POOL_BLOCK_SIZE_N / POOL_THREAD_BLOCK_N;
constexpr auto POOL_THREAD_SIZE_M = POOL_BLOCK_SIZE_M / POOL_THREAD_BLOCK_M;
constexpr auto POOL_THREAD_SIZE_K = POOL_BLOCK_DEPTH / POOL_THREAD_BLOCK_K;
constexpr auto MAX_POOL_SIZE = 8;

//function to calculate a single filter dot product in a single thread
template <int BLOCK_N, int BLOCK_M, int DEPTH>
__device__ float calculate_conv2d_dot(volatile float * s_filter, volatile float * s_load_block, int start_n, int start_m);

//forward pass function to convolve a filter over a set of input images and write the output to the "d_output" parameter
//the filter will be dot producted over each region of the image
void filter_convolve_2d(float * d_input, float * d_filter, float * d_output, shape input_shape, shape output_shape, shape filter_shape, shape padding, size_t batch_size);

//forward pass function to convolve a max pool over the input images
//each pool will take the maximum value and then stride to the next region of the image
//an array of maximum values will be returned. A mask variable is also create holding the indices from which each output came
void max_pool_2d(float * d_input, int * d_mask, float * d_output, shape input_shape, shape pool_size, shape stride, shape output_shape, shape padding, size_t batch_size);

//backward pass for the convolutional filter to get to the previous layer of partial derivatives
void filter_outer_convolve_2d(float * d_input, float * d_filter, float * d_output, shape input_shape, shape output_shape, shape filter_shape, shape padding, size_t batch_size);

//training derivative for the convolutional filter. Returns a set of derivatives with the same shape as the filter
void filter_convolve_2d_derivative(float * d_input, float * d_pds, float * d_output, shape input_shape, shape pd_shape, shape output_shape, shape padding, size_t batch_size);

//backward pass for the max pooling layer to get the previous layer of partial derivatives
void max_pool_2d_derivative(float * d_input, int * d_mask, float * d_output, shape input_shape, shape output_shape, size_t batch_size);
