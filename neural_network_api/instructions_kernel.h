#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <conio.h>
#include <algorithm>

#include <stdio.h>
#include <stdexcept>
#include <time.h>
//#include <math.h>
#include <curand.h>
#include <vector>

#include "kernel.h"

#include "timer.h"

using namespace std;

/*#define cuda_safe_call(err) __cuda_safe_call(err, __FILE__, __LINE__)

inline void __cuda_safe_call(cudaError error, char *file, int line, bool abort=true) {
	if (error != cudaSuccess) {
		printf("cuda_safe_call failed in file %s on line %d with error %s\n", file, line, cudaGetErrorString(error));
		if (abort)
			exit(error);
	}
}*/

constexpr auto MATMUL_BLOCK_SIZE = 64;
constexpr auto MATMUL_BLOCK_SIZEx2 = 128;
constexpr auto MAT_ROWS_PER_THREAD = 8;
constexpr auto INPUT_CHUNK_SIZE = 8;
constexpr auto WEIGHT_LOAD_SIZE = 1;
constexpr auto OUTPUTS_PER_THREAD = 1;
constexpr auto AVERAGE_CHUNK_SIZE = 16;
constexpr auto M_TILE_SIZE_X = 64;
constexpr auto M_TILE_SIZE_Y = 16;

template <unsigned int block_size>
__device__ void warp_reduce(volatile float * s_cost, int tid);
template <unsigned int block_size>
__device__ void warp_reduce_to_zero(volatile float * s_cost);

void allocate_device_pointer(float ** d_pointer, int size);
void deallocate_device_pointer(float * d_pointer);

void load_data_into_device(float * input_data, float * d_data_p, int size);
void retrieve_output_data(float * output_data, float * d_data_p, int size);
void copy_into_device_array(float * input_data, float * d_data_p, int size, int offset);
void copy_staggered_into_device_array(float * input_data, float * d_data_pointer, int in_dat_size, int out_dat_size, int num);

void get_prng(curandGenerator_t * prng, int seed);
void random_host_array(curandGenerator_t prng, float * array_p, float scale, float offset, int size, int seed);
//void fill_array(float * array_p, float value, int size);
void fill_device_array(float * d_array_p, float value, int size);

void add_matrices(float * d_input_p, float * d_out, float * d_bias_p, int size, int num);
//void multiply_matrices(float * d_input_p, float * d_out, float * d_partial_outputs, float * d_weights_p, int rows, int cols, int num);
void multiply_matrices(float * d_A, float * d_B, float * d_partial_outputs, float * d_out, int A_rows, int A_cols, int B_rows, int B_cols);
void multiply_staggered(float * d_input_p, float * d_out, float * d_partial_outputs, float * d_mul_mat, int rows, int cols, int stagger_size, int num);
void apply_relu(float * d_input_p, float * d_output_p, int size, float alpha);
void apply_softmax(float * d_input_p, float * d_output_p, int input_size, int num, float beta);
void relu_derivative(float * d_input_p, float * d_output_p, int size, float alpha);
//void softmax_derivative(float * d_input_p, float * d_output_p, int size, float beta);

void batch_norm(float * d_input_p, float * d_output_p, int size, int num);
void batch_norm_derivative(float * d_input_p, float * d_output_p, int size, int num);

void hadamard_product(float * d_a, float * d_b, float * d_output_p, int in_size);
void transpose(float * d_matrix_p, float * d_output_p, int rows, int cols);
//void distributive_hadamard_transpose(float * d_input_p, float * d_matrix_p, float * d_output_p, int mat_rows, int mat_colse, int cols, int vector_size, int num);

void average_vector(float * d_matrix, float * d_output_p, int size, int num, int divisor);
template <typename T_i, typename T_o>
extern void scalar_matrix_multiply(T_i * d_matrix, T_o * d_output_p, float scalar, int size);

void average_value(float * d_input_p, float * average, int size);
void average_value(float * d_input_p, float * average, int size, float divisor);

void subtract_partial_derivatives(float * d_matrix, float * d_derivatives, int size, float learning_rate);

void squared_error_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size);
void squared_error_cost_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size);

void softmax_cross_entropy_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num);
void softmax_cross_entropy_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num);
