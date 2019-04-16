#pragma once

#include "float_ops.h"
#include "d_matrix.h"
#include "error_handling.h"

//define constants used in matrix multiplication

//Block Tile Size
//Size of the matrix block computed by each thread block
constexpr auto BLOCK_TILE_N = 128;
constexpr auto BLOCK_TILE_M = 32;
constexpr auto BLOCK_TILE_K = 16;

//Thread Tile Size
//Size of the matrix block computed by each thread
constexpr auto THREAD_TILE_N = 8;
constexpr auto THREAD_TILE_M = 8;
constexpr auto THREAD_TILE_K = 2;

//Thread Sizes
//Number of threads in each dimension the kernel is launched with
constexpr auto THREAD_SIZE_N = BLOCK_TILE_N / THREAD_TILE_N;
constexpr auto THREAD_SIZE_M = BLOCK_TILE_M / THREAD_TILE_M;

//API FUNCTION
//Thread Level Multiply
//Multiplies two matrix fragments in a single thread
template <typename m_T, int tile_n, int tile_m, int tile_k>
__device__ void d_thread_level_multiply(m_T r_accum[tile_n * tile_m], m_T r_A[tile_m * tile_k], m_T r_B[tile_n * tile_k]);

namespace nnet {
	namespace nnet_internal {
		//API FUNCTION
		//Matrix Multiply
		//Multiplies two matrices A and B together and writes into matrix C
		//Uses the GEMM matrix multiplication chunking concept
		template <typename m_T, mat_order A_o = mat_order::MAT_ORDER_ROW, mat_order B_o = mat_order::MAT_ORDER_ROW, mat_order C_o = mat_order::MAT_ORDER_ROW>
		void matrix_multiply(d_matrix<m_T> & A, d_matrix<m_T> & B, d_matrix<m_T> & C);
	}
}