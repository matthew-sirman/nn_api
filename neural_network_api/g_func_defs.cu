#pragma once

#include "matrix_multiply.h"

#include <stdexcept>

using namespace std;

using namespace nnet::util;

namespace nnet {
	namespace nnet_internal {
		//matrix_multiply definitions

		//kernel function to multiply two matrices together
		template <typename m_T, mat_order A_o = mat_order::MAT_ORDER_ROW, mat_order B_o = mat_order::MAT_ORDER_ROW, mat_order C_o = mat_order::MAT_ORDER_ROW>
		__global__ void d_matrix_multiply(m_T * A, m_T * B, m_T * C, int n_dim, int m_dim, int k_dim) {
			//This algorithm essentially divides the matrix up into different sized blocks for
			//different sized thread structures, and breaks it down into thread level fragments.
			//Computes the individual products at thread level and then steps back up to global
			//level to reconstruct a matrix as the result.

			//allocate block tiles from each matrix
			__shared__ m_T s_A_tile[BLOCK_TILE_M * BLOCK_TILE_K];
			__shared__ m_T s_B_tile[BLOCK_TILE_K * BLOCK_TILE_N];

			//allocate an accumulator for the output chunk
			__shared__ m_T s_accumulator[BLOCK_TILE_M * BLOCK_TILE_N];

			//allocate fragment tiles in registers
			m_T r_A_frag[THREAD_TILE_M * THREAD_TILE_K];
			m_T r_B_frag[THREAD_TILE_N * THREAD_TILE_K];

			//allocate fragment accumulator in registers
			m_T r_accumulator[THREAD_TILE_M * THREAD_TILE_N];

			//get the index of the current tile
			unsigned int tile_n = blockIdx.x * BLOCK_TILE_N;
			unsigned int tile_m = blockIdx.y * BLOCK_TILE_M;

			//get the index of the current thread position
			unsigned int n_id = threadIdx.x;
			unsigned int m_id = threadIdx.y;

			//get the linear index of the current thread
			unsigned int t_id = n_id + THREAD_SIZE_N * m_id;

			//fill the shared accumulator with 0s to avoid potential
			//memory issues from previously stored data
#pragma unroll
			for (int m = 0; m < THREAD_TILE_M; m++) {
#pragma unroll
				for (int n = 0; n < THREAD_TILE_N; n++) {
					s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N + n_id * THREAD_TILE_N + n] = m_T();
				}
			}

			//synchronise to ensure each thread has finished clearing the accumulator
			cuda_syncthreads();

			//stride through the K dimension (equal dimension of the two matrices)
			for (int bk = 0; bk < k_dim; bk += BLOCK_TILE_K) {
				//load tiles into shared memory

				//load the tile at position K from A that corresponds to the current tile (tile_n, tile_m)
#pragma unroll
				for (int load = 0; load < BLOCK_TILE_M * BLOCK_TILE_K; load += THREAD_SIZE_M * THREAD_SIZE_N) {
					int load_id = load + t_id;
					if (load_id < BLOCK_TILE_M * BLOCK_TILE_K) {
						//get the local shared index
						int load_m = load_id / BLOCK_TILE_K;
						int load_k = load_id % BLOCK_TILE_K;

						//get the global load index
						int g_load_m = tile_m + load_m;
						int g_load_k = bk + load_k;

						//if the load is within range
						if (g_load_m < m_dim && g_load_k < k_dim) {
							//load from the correct location, depending whether the matrix is in row or column major
							//order
							if (A_o == mat_order::MAT_ORDER_ROW) {
								s_A_tile[load_m * BLOCK_TILE_K + load_k] = A[g_load_m * k_dim + g_load_k];
							}
							else {
								s_A_tile[load_m * BLOCK_TILE_K + load_k] = A[g_load_k * m_dim + g_load_m];
							}
						}
						//otherwise set to 0
						else {
							s_A_tile[load_m * BLOCK_TILE_K + load_k + 0] = m_T();
						}
					}
				}

				//load the tile at position K from B that corresponds to the current tile (tile_n, tile_m)
#pragma unroll
				for (int load = 0; load < BLOCK_TILE_N * BLOCK_TILE_K; load += THREAD_SIZE_M * THREAD_SIZE_N * 4) {
					int load_id = load + t_id * 4;
					if (load_id < BLOCK_TILE_K * BLOCK_TILE_N) {
						//get the local shared index
						int load_n = load_id % BLOCK_TILE_N;
						int load_k = load_id / BLOCK_TILE_N;

						//get the global load index
						int g_load_n = tile_n + load_n;
						int g_load_k = bk + load_k;

						//load 4 elements from global memory to shared memory
						//(through extensive testing I found this to give the quickest load
						//speed)
						//Each "if" case is a different permutation of the number of elements
						//loaded from this thread, and matrix order is taken into account
						if (g_load_n + 3 < n_dim && g_load_k < k_dim) {
							if (B_o == mat_order::MAT_ORDER_ROW) {
								reinterpret_cast<float4*>(s_B_tile)[(load_k * BLOCK_TILE_N + load_n) / 4] = reinterpret_cast<float4*>(B)[(g_load_k * n_dim + g_load_n) / 4];
							}
							else {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[(g_load_n + 0) * k_dim + g_load_k];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = B[(g_load_n + 1) * k_dim + g_load_k];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 2] = B[(g_load_n + 2) * k_dim + g_load_k];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 3] = B[(g_load_n + 3) * k_dim + g_load_k];
							}
						}
						else if (g_load_n + 2 < n_dim && g_load_k < k_dim) {
							if (B_o == mat_order::MAT_ORDER_ROW) {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[g_load_k * n_dim + g_load_n + 0];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = B[g_load_k * n_dim + g_load_n + 1];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 2] = B[g_load_k * n_dim + g_load_n + 2];
							}
							else {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[(g_load_n + 0) * k_dim + g_load_k];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = B[(g_load_n + 1) * k_dim + g_load_k];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 2] = B[(g_load_n + 2) * k_dim + g_load_k];
							}
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 3] = m_T();
						}
						else if (g_load_n + 1 < n_dim && g_load_k < k_dim) {
							if (B_o == mat_order::MAT_ORDER_ROW) {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[g_load_k * n_dim + g_load_n + 0];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = B[g_load_k * n_dim + g_load_n + 1];
							}
							else {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[(g_load_n + 0) * k_dim + g_load_k];
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = B[(g_load_n + 1) * k_dim + g_load_k];
							}
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 2] = m_T();
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 3] = m_T();
						}
						else if (g_load_n + 0 < n_dim && g_load_k < k_dim) {
							if (B_o == mat_order::MAT_ORDER_ROW) {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[g_load_k * n_dim + g_load_n + 0];
							}
							else {
								s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = B[(g_load_n + 0) * k_dim + g_load_k];
							}
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = m_T();
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 2] = m_T();
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 3] = m_T();
						}
						else {
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 0] = m_T();
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 1] = m_T();
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 2] = m_T();
							s_B_tile[load_k * BLOCK_TILE_N + load_n + 3] = m_T();
						}
					}
				}

				//synchronise to complete loading before moving on
				cuda_syncthreads();

				//now calculate the product

				//stride through the K dimension in the tile
#pragma unroll
				for (int wk = 0; wk < BLOCK_TILE_K; wk += THREAD_TILE_K) {
					//load the fragment from shared memory on a single thread
#pragma unroll
					for (int load_k = 0; load_k < THREAD_TILE_K; load_k++) {
#pragma unroll
						for (int load_m = 0; load_m < THREAD_TILE_M; load_m++) {
							r_A_frag[load_m * THREAD_TILE_K + load_k] = s_A_tile[(m_id * THREAD_TILE_M + load_m) * BLOCK_TILE_K + wk + load_k];
						}
#pragma unroll
						for (int load_n = 0; load_n < THREAD_TILE_N; load_n++) {
							r_B_frag[load_k * THREAD_TILE_N + load_n] = s_B_tile[(wk + load_k) * BLOCK_TILE_N + n_id * THREAD_TILE_N + load_n];
						}
					}

					//set the accumulator to 0 to avoid memory issues
#pragma unroll
					for (int fill = 0; fill < THREAD_TILE_N * THREAD_TILE_M; fill++) {
						r_accumulator[fill] = 0;
					}

					//calculate the dot product between the two fragments and write to the accumulator
					d_thread_level_multiply<m_T, THREAD_TILE_N, THREAD_TILE_M, THREAD_TILE_K>(r_accumulator, r_A_frag, r_B_frag);

					//copy the fragment into the shared accumulator
#pragma unroll
					for (int m = 0; m < THREAD_TILE_M; m++) {
#pragma unroll
						for (int n = 0; n < THREAD_TILE_N; n++) {
							s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N + n_id * THREAD_TILE_N + n] += r_accumulator[m * THREAD_TILE_N + n];
						}
					}
					cuda_syncthreads();
				}
			}

			//write the accumulated values to global memory
			//writes are shared among all the threads for optimisation
#pragma unroll
			for (int write = 0; write < BLOCK_TILE_M * BLOCK_TILE_N; write += THREAD_SIZE_M * THREAD_SIZE_N) {
				int write_id = write + t_id;
				if (write_id < BLOCK_TILE_M * BLOCK_TILE_N) {
					//get the output index in shared memory
					int write_n = write_id % BLOCK_TILE_N;
					int write_m = write_id / BLOCK_TILE_N;

					//get the output index in global memory
					int g_write_n = tile_n + write_n;
					int g_write_m = tile_m + write_m;

					//check if the output index is in range
					if (g_write_n < n_dim && g_write_m < m_dim) {
						//write to the correct location in memory depending on the matrix order of the output
						if (C_o == mat_order::MAT_ORDER_ROW) {
							C[g_write_m * n_dim + g_write_n] = s_accumulator[write_m * BLOCK_TILE_N + write_n];
						}
						else {
							C[g_write_n * m_dim + g_write_m] = s_accumulator[write_m * BLOCK_TILE_N + write_n];
						}
					}
				}
			}
		}

		template <typename m_T, int tile_n, int tile_m, int tile_k>
		__device__ void d_thread_level_multiply(m_T r_accum[tile_n * tile_m], m_T r_A[tile_m * tile_k], m_T r_B[tile_n * tile_k]) {
			//loop through each dimension of the fragmenst and increment the acculuator with the product
#pragma unroll
			for (int t_k = 0; t_k < tile_k; t_k++) {
#pragma unroll
				for (int t_m = 0; t_m < tile_m; t_m++) {
					//load and cache this value to minimise loads (even though 
					//the loads are from registers, reduces indexing operations)
					m_T tmp = r_A[t_m * tile_k + t_k];
#pragma unroll
					for (int t_n = 0; t_n < tile_n; t_n++)
						r_accum[t_m * tile_m + t_n] += tmp * r_B[t_k * tile_n + t_n];
				}
			}
		}

		template <typename m_T, mat_order A_o, mat_order B_o, mat_order C_o>
		void matrix_multiply(d_matrix<m_T> & A, d_matrix<m_T> & B, d_matrix<m_T> & C)
		{
			//check the K dimensions line up, otherwise matrix multiplication is impossible
			ERR_ASSERT(A.n_size != B.m_size, "Cannot multiply matricies with different K dimensions");

			//setup thread and block sizes from constants
			dim3 threads_per_block(THREAD_SIZE_N, THREAD_SIZE_M);
			dim3 blocks_per_grid(ceil_div(BLOCK_TILE_N, B.n_size), ceil_div(BLOCK_TILE_M, A.m_size));

			//launch the multiply kernel
			d_matrix_multiply<m_T, A_o, B_o, C_o> << <blocks_per_grid, threads_per_block >> > (A.d_data, B.d_data, C.d_data, B.n_size, A.m_size, A.n_size);
		}

		//instantiate templates for each permutation of matrix orders to avoid unresolved
		//externals, as the source files are compiled at runtime so the template source code is lost
		//Essentially, as this function is not defined in a header, it must be specifically instantiated
		//to work, and as it uses the special Cuda launch syntax (<<<>>>), it cannit be defined in a header
		//for full generalisation.

		template void matrix_multiply<float, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_ROW>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_COL>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_ROW>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_ROW>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_ROW, mat_order::MAT_ORDER_COL>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_ROW>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
		template void matrix_multiply<float, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL, mat_order::MAT_ORDER_COL>(d_matrix<float>&, d_matrix<float>&, d_matrix<float>&);
	}
}