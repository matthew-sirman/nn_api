#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#include "kernel.h"

#include "shape.h"

//constexpr auto MAX_FILTER_SIZE = 16;
//constexpr auto MAX_FILTER_DEPTH = 8;

constexpr auto CONV_BLOCK_SIZE_N = 32;
constexpr auto CONV_BLOCK_SIZE_M = 16;
constexpr auto CONV_BLOCK_SIZE_K = 2;
constexpr auto CONV_THREAD_BLOCK_N = 4;
constexpr auto CONV_THREAD_BLOCK_M = 2;
constexpr auto CONV_THREAD_BLOCK_K = 2;
constexpr auto CONV_BATCH_DEPTH = 1;
constexpr auto CONV_THREAD_SIZE_N = CONV_BLOCK_SIZE_N / CONV_THREAD_BLOCK_N;
constexpr auto CONV_THREAD_SIZE_M = CONV_BLOCK_SIZE_M / CONV_THREAD_BLOCK_M;
constexpr auto CONV_THREAD_SIZE_K = CONV_BLOCK_SIZE_K / CONV_THREAD_BLOCK_K;

constexpr auto FILTER_BLOCK_SIZE_N = 8;
constexpr auto FILTER_BLOCK_SIZE_M = 8;
constexpr auto FILTER_BLOCK_SIZE_K = CONV_BLOCK_SIZE_K;
constexpr auto FILTER_THREAD_BLOCK_N = FILTER_BLOCK_SIZE_N / CONV_THREAD_SIZE_N;
constexpr auto FILTER_THREAD_BLOCK_M = FILTER_BLOCK_SIZE_M / CONV_THREAD_SIZE_M;

constexpr auto CONV_OUTER_BLOCK_SIZE_N = 32;
constexpr auto CONV_OUTER_BLOCK_SIZE_M = 16;
constexpr auto CONV_OUTER_BLOCK_SIZE_K = 1;
constexpr auto CONV_OUTER_THREAD_BLOCK_N = 4;
constexpr auto CONV_OUTER_THREAD_BLOCK_M = 4;
constexpr auto CONV_OUTER_THREAD_BLOCK_K = 1;
constexpr auto CONV_OUTER_THREAD_SIZE_N = CONV_OUTER_BLOCK_SIZE_N / CONV_OUTER_THREAD_BLOCK_N;
constexpr auto CONV_OUTER_THREAD_SIZE_M = CONV_OUTER_BLOCK_SIZE_M / CONV_OUTER_THREAD_BLOCK_M;
constexpr auto CONV_OUTER_THREAD_SIZE_K = CONV_OUTER_BLOCK_SIZE_K / CONV_OUTER_THREAD_BLOCK_K;

constexpr auto CONV_DER_BLOCK_SIZE_N = 32;
constexpr auto CONV_DER_BLOCK_SIZE_M = 32;
constexpr auto CONV_DER_BLOCK_SIZE_K = 2;
constexpr auto CONV_DER_THREAD_BLOCK_N = 4;
constexpr auto CONV_DER_THREAD_BLOCK_M = 4;
constexpr auto CONV_DER_THREAD_BLOCK_K = 2;
constexpr auto CONV_DER_THREAD_SIZE_N = CONV_DER_BLOCK_SIZE_N / CONV_DER_THREAD_BLOCK_N;
constexpr auto CONV_DER_THREAD_SIZE_M = CONV_DER_BLOCK_SIZE_M / CONV_DER_THREAD_BLOCK_M;
constexpr auto CONV_DER_THREAD_SIZE_K = CONV_DER_BLOCK_SIZE_K / CONV_DER_THREAD_BLOCK_K;

/*
constexpr auto CONV_EXP_BLOCK_SIZE_N = 16;
constexpr auto CONV_EXP_BLOCK_SIZE_M = 16;
constexpr auto CONV_EXP_THREAD_BLOCK_N = 1;
constexpr auto CONV_EXP_THREAD_BLOCK_M = 1;
constexpr auto CONV_EXP_THREAD_BLOCK_K = 2;
constexpr auto CONV_EXP_THREAD_SIZE_N = CONV_EXP_BLOCK_SIZE_N / CONV_EXP_THREAD_BLOCK_N;
constexpr auto CONV_EXP_THREAD_SIZE_M = CONV_EXP_BLOCK_SIZE_M / CONV_EXP_THREAD_BLOCK_M;
constexpr auto CONV_EXP_THREAD_SIZE_K = MAX_FILTER_DEPTH / CONV_EXP_THREAD_BLOCK_K;
*/

/*
constexpr auto CONV_DER_THREAD_SIZE_N = 32;
constexpr auto CONV_DER_THREAD_SIZE_M = 16;
constexpr auto CONV_DER_THREAD_SIZE_K = 1;
constexpr auto CONV_DER_THREAD_SIZE = CONV_DER_THREAD_SIZE_N * CONV_DER_THREAD_SIZE_M * CONV_DER_THREAD_SIZE_K;
*/

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

template <int BLOCK_N, int BLOCK_M, int DEPTH>
__device__ float calculate_conv2d_dot(volatile float * s_filter, volatile float * s_load_block, int start_n, int start_m);

void filter_convolve_2d(float * d_input, float * d_filter, float * d_output, shape input_shape, shape output_shape, shape filter_shape, shape padding, size_t batch_size);
void max_pool_2d(float * d_input, int * d_mask, float * d_output, shape input_shape, shape pool_size, shape stride, shape output_shape, size_t batch_size);

void filter_outer_convolve_2d(float * d_input, float * d_filter, float * d_output, shape input_shape, shape output_shape, shape filter_shape, shape padding, size_t batch_size);
void filter_convolve_2d_derivative(float * d_input, float * d_pds, float * d_output, shape input_shape, shape pd_shape, shape output_shape, shape padding, size_t batch_size);
void max_pool_2d_derivative(float * d_input, int * d_mask, float * d_output, shape input_shape, shape output_shape, size_t batch_size);
