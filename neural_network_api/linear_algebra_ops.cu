//#include "linear_algebra_ops.h"

//__global__ void d_matrix_multiply(float * A, float * B, float * out, int n_dim, int m_dim, int k_dim) {

/*template <typename m_T>
__global__ void d_matrix_multiply(d_matrix<m_T> &A, d_matrix<m_T> &B, d_matrix<m_T> &C) {
	__shared__ m_T s_A_tile[BLOCK_TILE_M][BLOCK_TILE_K];
	__shared__ m_T s_B_tile[BLOCK_TILE_K][BLOCK_TILE_N];

	__shared__ m_T s_accumulator[BLOCK_TILE_M * BLOCK_TILE_N];

	m_T r_A_frag[THREAD_TILE_M * THREAD_TILE_K];
	m_T r_B_frag[THREAD_TILE_N * THREAD_TILE_K];

	m_T r_accumulator[THREAD_TILE_M * THREAD_TILE_N];

	int n_dim = B.n_size;
	int m_dim = A.m_size;
	int k_dim = A.n_size;

	unsigned int tile_n = blockIdx.x * BLOCK_TILE_N;
	unsigned int tile_m = blockIdx.y * BLOCK_TILE_M;

	unsigned int n_id = threadIdx.x;
	unsigned int m_id = threadIdx.y;

	unsigned int t_id = n_id * THREAD_SIZE_M + m_id;

	#pragma unroll
	for (int m = 0; m < THREAD_TILE_M; m++) {
		#pragma unroll
		for (int n = 0; n < THREAD_TILE_N; n++) {
			s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N + n_id * THREAD_TILE_N + n] = m_T();
		}
	}

	cuda_syncthreads();

	for (int bk = 0; bk < k_dim; bk += BLOCK_TILE_K) {
		//load tiles into shared memory
		
		#pragma unroll
		for (int load = 0; load < BLOCK_TILE_M * BLOCK_TILE_K; load += THREAD_SIZE_M * THREAD_SIZE_N) {
			int load_id = load + t_id;
			if (load_id < BLOCK_TILE_M * BLOCK_TILE_K) {
				int load_m = load_id / BLOCK_TILE_K;
				int load_k = load_id % BLOCK_TILE_K;

				int g_load_m = tile_m + load_m;
				int g_load_k = bk + load_k;

				if (g_load_m < m_dim && g_load_k < k_dim) {
					s_A_tile[load_m][load_k] = A.get(g_load_k, g_load_m);
					//s_A_tile[load_m][load_k] = A.d_data[g_load_m * k_dim + g_load_k];
				}
				else {
					s_A_tile[load_m][load_k] = m_T();
				}
			}
		}

		#pragma unroll
		for (int load_n = 0; load_n < BLOCK_TILE_N; load_n += THREAD_SIZE_N) {
			#pragma unroll (BLOCK_TILE_K / THREAD_SIZE_M)
			for (int load_k = 0; load_k < BLOCK_TILE_K; load_k += THREAD_SIZE_M) {
				if (load_k + m_id < BLOCK_TILE_K && load_n + n_id < BLOCK_TILE_N) {
					if (tile_n + load_n + n_id < n_dim && bk + load_k + m_id < k_dim) {
						//s_B_tile[(load_k + m_id) * BLOCK_TILE_N + load_n + n_id] = B[n_dim * (bk + load_k + m_id) + tile_n + load_n + n_id];
						//s_B_tile[load_k + m_id][load_n + n_id] = B.d_data[n_dim * (bk + load_k + m_id) + tile_n + load_n + n_id];
						s_B_tile[load_k + m_id][load_n + n_id] = B.get(tile_n + load_n + n_id, bk + load_k + m_id);
					}
					else {
						//s_B_tile[(load_k + m_id) * BLOCK_TILE_N + load_n + n_id] = 0.0;
						s_B_tile[load_k + m_id][load_n + n_id] = m_T();
					}
				}
			}
		}

		cuda_syncthreads();

		#pragma unroll
		for (int wk = 0; wk < BLOCK_TILE_K; wk += THREAD_TILE_K) {
			//load fragments into registers

			#pragma unroll
			for (int load_k = 0; load_k < THREAD_TILE_K; load_k++) {
				#pragma unroll
				for (int load_m = 0; load_m < THREAD_TILE_M; load_m++) {
					r_A_frag[load_m * THREAD_TILE_K + load_k] = s_A_tile[m_id * THREAD_TILE_M + load_m][wk + load_k];
				}
				#pragma unroll
				for (int load_n = 0; load_n < THREAD_TILE_N; load_n++) {
					r_B_frag[load_k * THREAD_TILE_N + load_n] = s_B_tile[wk + load_k][n_id * THREAD_TILE_N + load_n];
				}
			}

			#pragma unroll
			for (int fill = 0; fill < THREAD_TILE_N * THREAD_TILE_M; fill++) {
				r_accumulator[fill] = 0;
			}

			d_thread_level_multiply<m_T, THREAD_TILE_N, THREAD_TILE_M, THREAD_TILE_K>(r_accumulator, r_A_frag, r_B_frag);

			#pragma unroll
			for (int m = 0; m < THREAD_TILE_M; m++) {
				#pragma unroll
				for (int n = 0; n < THREAD_TILE_N; n++) {
					s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N + n_id * THREAD_TILE_N + n] += r_accumulator[m * THREAD_TILE_N + n];
				}
			}
		}
		cuda_syncthreads();
	}

	#pragma unroll
	for (int write = 0; write < BLOCK_TILE_M * BLOCK_TILE_N; write += THREAD_SIZE_M * THREAD_SIZE_N) {
		int write_id = write + t_id;
		if (write_id < BLOCK_TILE_M * BLOCK_TILE_N) {
			int write_n = write_id % BLOCK_TILE_N;
			int write_m = write_id / BLOCK_TILE_N;

			int g_write_n = tile_n + write_n;
			int g_write_m = tile_m + write_m;

			if (g_write_n < n_dim && g_write_m < m_dim) {
				C.d_data[g_write_m * n_dim + g_write_n] = s_accumulator[write_m * BLOCK_TILE_N + write_n];
				//C.incr(g_write_n, g_write_m, s_accumulator[write_m * BLOCK_TILE_N + write_n]);
			}
		}
	}
}

template <>
__global__ void d_matrix_multiply(d_matrix<float2> &A, d_matrix<float2> &B, d_matrix<float2> &C) {
	__shared__ float2 s_A_tile[BLOCK_TILE_M][BLOCK_TILE_K / 2];
	__shared__ float2 s_B_tile[BLOCK_TILE_K][BLOCK_TILE_N / 2];

	__shared__ float2 s_accumulator[BLOCK_TILE_M * BLOCK_TILE_N / 2];

	float2 r_A_frag[THREAD_TILE_M * THREAD_TILE_K / 2];
	float2 r_B_frag[THREAD_TILE_N * THREAD_TILE_K / 2];

	float2 r_accumulator[THREAD_TILE_M * THREAD_TILE_N / 2];

	int n_dim = B.n_size;
	int m_dim = A.m_size;
	int k_dim = B.m_size;

	unsigned int tile_n = blockIdx.x * BLOCK_TILE_N / 2;
	unsigned int tile_m = blockIdx.y * BLOCK_TILE_M;

	unsigned int n_id = threadIdx.x;
	unsigned int m_id = threadIdx.y;

	unsigned int t_id = n_id * THREAD_SIZE_M + m_id;

	#pragma unroll
	for (int m = 0; m < THREAD_TILE_M; m++) {
		#pragma unroll
		for (int n = 0; n < THREAD_TILE_N / 2; n++) {
			s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N / 2 + n_id * THREAD_TILE_N / 2 + n] = { 0.0, 0.0 };
		}
	}

	cuda_syncthreads();

	for (int bk = 0; bk < k_dim; bk += BLOCK_TILE_K) {
		//load tiles into shared memory
		
		#pragma unroll
		for (int load = 0; load < BLOCK_TILE_M * BLOCK_TILE_K / 2; load += THREAD_SIZE_M * THREAD_SIZE_N) {
			int load_id = load + t_id;
			if (load_id < BLOCK_TILE_M * BLOCK_TILE_K / 2) {
				int load_m = load_id / (BLOCK_TILE_K / 2);
				int load_k = load_id % (BLOCK_TILE_K / 2);

				int g_load_m = tile_m + load_m;
				int g_load_k = bk / 2 + load_k;

				if (g_load_m < m_dim && g_load_k < k_dim) {
					s_A_tile[load_m][load_k] = A.get(g_load_k, g_load_m);
					//s_A_tile[load_m][load_k] = A.d_data[g_load_m * k_dim + g_load_k];
				}
				else {
					s_A_tile[load_m][load_k] = { 0.0, 0.0 };
				}
			}
		}

		#pragma unroll
		for (int load_n = 0; load_n < BLOCK_TILE_N / 2; load_n += THREAD_SIZE_N) {
			#pragma unroll
			for (int load_k = 0; load_k < BLOCK_TILE_K; load_k += THREAD_SIZE_M) {
				if (load_k + m_id < BLOCK_TILE_K && load_n + n_id < BLOCK_TILE_N / 2) {
					if (tile_n + load_n + n_id < n_dim && bk + load_k + m_id < k_dim) {
						//s_B_tile[(load_k + m_id) * BLOCK_TILE_N + load_n + n_id] = B[n_dim * (bk + load_k + m_id) + tile_n + load_n + n_id];
						//s_B_tile[load_k + m_id][load_n + n_id] = B.d_data[n_dim * (bk + load_k + m_id) + tile_n + load_n + n_id];
						s_B_tile[load_k + m_id][load_n + n_id] = B.get(tile_n + load_n + n_id, bk + load_k + m_id);
					}
					else {
						//s_B_tile[(load_k + m_id) * BLOCK_TILE_N + load_n + n_id] = 0.0;
						s_B_tile[load_k + m_id][load_n + n_id] = { 0.0, 0.0 };
					}
				}
			}
		}

		cuda_syncthreads();

		#pragma unroll
		for (int wk = 0; wk < BLOCK_TILE_K; wk += THREAD_TILE_K) {
			//load fragments into registers

			#pragma unroll
			for (int load_k = 0; load_k < THREAD_TILE_K / 2; load_k++) {
				#pragma unroll
				for (int load_m = 0; load_m < THREAD_TILE_M; load_m++) {
					r_A_frag[load_m * THREAD_TILE_K / 2 + load_k] = s_A_tile[m_id * THREAD_TILE_M + load_m][wk / 2 + load_k];
				}
				#pragma unroll
				for (int load_n = 0; load_n < THREAD_TILE_N / 2; load_n++) {
					r_B_frag[load_k * THREAD_TILE_N + load_n] = s_B_tile[wk + 2 * load_k][n_id * THREAD_TILE_N / 2 + load_n];
					r_B_frag[load_k * THREAD_TILE_N + THREAD_TILE_N / 2 + load_n] = s_B_tile[wk + 2 * load_k + 1][n_id * THREAD_TILE_N / 2 + load_n];
				}
			}

			#pragma unroll
			for (int fill = 0; fill < THREAD_TILE_N * THREAD_TILE_M / 2; fill++) {
				r_accumulator[fill] = { 0.0, 0.0 };
			}

			d_thread_level_multiply<float, THREAD_TILE_N, THREAD_TILE_M, THREAD_TILE_K>(
				reinterpret_cast<float*>(r_accumulator), 
				reinterpret_cast<float*>(r_A_frag), 
				reinterpret_cast<float*>(r_B_frag));

			#pragma unroll
			for (int m = 0; m < THREAD_TILE_M; m++) {
				#pragma unroll
				for (int n = 0; n < THREAD_TILE_N / 2; n++) {
					s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N / 2 + n_id * THREAD_TILE_N / 2 + n] += r_accumulator[m * THREAD_TILE_N / 2 + n];
				}
			}
		}
		cuda_syncthreads();
	}

	#pragma unroll
	for (int write = 0; write < BLOCK_TILE_M * BLOCK_TILE_N / 2; write += THREAD_SIZE_M * THREAD_SIZE_N) {
		int write_id = write + t_id;
		if (write_id < BLOCK_TILE_M * BLOCK_TILE_N / 2) {
			int write_n = write_id % (BLOCK_TILE_N / 2);
			int write_m = write_id / (BLOCK_TILE_N / 2);

			int g_write_n = tile_n + write_n;
			int g_write_m = tile_m + write_m;

			if (g_write_n < n_dim && g_write_m < m_dim) {
				C.d_data[g_write_m * n_dim + g_write_n] = s_accumulator[write_m * BLOCK_TILE_N / 2 + write_n];
				//C.incr(g_write_n, g_write_m, s_accumulator[write_m * BLOCK_TILE_N + write_n]);
			}
		}
	}
}

template <>
__global__ void d_matrix_multiply(d_matrix<float4> &A, d_matrix<float4> &B, d_matrix<float4> &C) {
	__shared__ float4 s_A_tile[BLOCK_TILE_M][BLOCK_TILE_K / 4];
	__shared__ float4 s_B_tile[BLOCK_TILE_K][BLOCK_TILE_N / 4];

	__shared__ float4 s_accumulator[BLOCK_TILE_M * BLOCK_TILE_N / 4];

	float4 r_A_frag[THREAD_TILE_M * THREAD_TILE_K / 4];
	float4 r_B_frag[THREAD_TILE_N * THREAD_TILE_K / 4];

	float4 r_accumulator[THREAD_TILE_M * THREAD_TILE_N / 4];

	int n_dim = B.n_size;
	int m_dim = A.m_size;
	int k_dim = B.m_size;

	unsigned int tile_n = blockIdx.x * BLOCK_TILE_N / 4;
	unsigned int tile_m = blockIdx.y * BLOCK_TILE_M;

	unsigned int n_id = threadIdx.x;
	unsigned int m_id = threadIdx.y;

	unsigned int t_id = n_id * THREAD_SIZE_M + m_id;

	#pragma unroll
	for (int m = 0; m < THREAD_TILE_M; m++) {
		#pragma unroll
		for (int n = 0; n < THREAD_TILE_N / 4; n++) {
			s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N / 4 + n_id * THREAD_TILE_N / 4 + n] = { 0.0, 0.0, 0.0, 0.0 };
		}
	}

	cuda_syncthreads();

	for (int bk = 0; bk < k_dim; bk += BLOCK_TILE_K) {
		//load tiles into shared memory
		
		#pragma unroll
		for (int load = 0; load < BLOCK_TILE_M * BLOCK_TILE_K / 4; load += THREAD_SIZE_M * THREAD_SIZE_N) {
			int load_id = load + t_id;
			if (load_id < BLOCK_TILE_M * BLOCK_TILE_K / 4) {
				int load_m = load_id / (BLOCK_TILE_K / 4);
				int load_k = load_id % (BLOCK_TILE_K / 4);

				int g_load_m = tile_m + load_m;
				int g_load_k = bk / 4 + load_k;

				if (g_load_m < m_dim && g_load_k < k_dim) {
					s_A_tile[load_m][load_k] = A.get(g_load_k, g_load_m);
					//s_A_tile[load_m][load_k] = A.d_data[g_load_m * k_dim + g_load_k];
				}
				else {
					s_A_tile[load_m][load_k] = { 0.0, 0.0, 0.0, 0.0 };
				}
			}
		}

		#pragma unroll
		for (int load_n = 0; load_n < BLOCK_TILE_N / 4; load_n += THREAD_SIZE_N) {
			#pragma unroll
			for (int load_k = 0; load_k < BLOCK_TILE_K; load_k += THREAD_SIZE_M) {
				if (load_k + m_id < BLOCK_TILE_K && load_n + n_id < BLOCK_TILE_N / 4) {
					if (tile_n + load_n + n_id < n_dim && bk + load_k + m_id < k_dim) {
						//s_B_tile[(load_k + m_id) * BLOCK_TILE_N + load_n + n_id] = B[n_dim * (bk + load_k + m_id) + tile_n + load_n + n_id];
						//s_B_tile[load_k + m_id][load_n + n_id] = B.d_data[n_dim * (bk + load_k + m_id) + tile_n + load_n + n_id];
						s_B_tile[load_k + m_id][load_n + n_id] = B.get(tile_n + load_n + n_id, bk + load_k + m_id);
					}
					else {
						//s_B_tile[(load_k + m_id) * BLOCK_TILE_N + load_n + n_id] = 0.0;
						s_B_tile[load_k + m_id][load_n + n_id] = { 0.0, 0.0, 0.0, 0.0 };
					}
				}
			}
		}

		cuda_syncthreads();

		#pragma unroll
		for (int wk = 0; wk < BLOCK_TILE_K; wk += THREAD_TILE_K) {
			//load fragments into registers

			#pragma unroll
			for (int load_k = 0; load_k < THREAD_TILE_K / 4; load_k++) {
				#pragma unroll
				for (int load_m = 0; load_m < THREAD_TILE_M; load_m++) {
					r_A_frag[load_m * THREAD_TILE_K / 4 + load_k] = s_A_tile[m_id * THREAD_TILE_M + load_m][wk / 4 + load_k];
				}
				#pragma unroll
				for (int load_n = 0; load_n < THREAD_TILE_N / 2; load_n++) {
					r_B_frag[load_k * THREAD_TILE_N + load_n] = s_B_tile[wk + 4 * load_k][n_id * THREAD_TILE_N / 4 + load_n];
					r_B_frag[load_k * THREAD_TILE_N + THREAD_TILE_N / 4 + load_n] = s_B_tile[wk + 4 * load_k + 1][n_id * THREAD_TILE_N / 4 + load_n];
					r_B_frag[load_k * THREAD_TILE_N + THREAD_TILE_N / 2 + load_n] = s_B_tile[wk + 4 * load_k + 2][n_id * THREAD_TILE_N / 4 + load_n];
					r_B_frag[load_k * THREAD_TILE_N + 3 * THREAD_TILE_N / 4 + load_n] = s_B_tile[wk + 4 * load_k + 3][n_id * THREAD_TILE_N / 4 + load_n];
				}
			}

			#pragma unroll
			for (int fill = 0; fill < THREAD_TILE_N * THREAD_TILE_M / 4; fill++) {
				r_accumulator[fill] = { 0.0, 0.0, 0.0, 0.0 };
			}

			d_thread_level_multiply<float, THREAD_TILE_N, THREAD_TILE_M, THREAD_TILE_K>(
				reinterpret_cast<float*>(r_accumulator), 
				reinterpret_cast<float*>(r_A_frag), 
				reinterpret_cast<float*>(r_B_frag));

			#pragma unroll
			for (int m = 0; m < THREAD_TILE_M; m++) {
				#pragma unroll
				for (int n = 0; n < THREAD_TILE_N / 4; n++) {
					s_accumulator[(m_id * THREAD_TILE_M + m) * BLOCK_TILE_N / 4 + n_id * THREAD_TILE_N / 4 + n] += r_accumulator[m * THREAD_TILE_N / 4 + n];
				}
			}
		}
		cuda_syncthreads();
	}

	#pragma unroll
	for (int write = 0; write < BLOCK_TILE_M * BLOCK_TILE_N / 4; write += THREAD_SIZE_M * THREAD_SIZE_N) {
		int write_id = write + t_id;
		if (write_id < BLOCK_TILE_M * BLOCK_TILE_N / 4) {
			int write_n = write_id % (BLOCK_TILE_N / 4);
			int write_m = write_id / (BLOCK_TILE_N / 4);

			int g_write_n = tile_n + write_n;
			int g_write_m = tile_m + write_m;

			if (g_write_n < n_dim && g_write_m < m_dim) {
				C.d_data[g_write_m * n_dim + g_write_n] = s_accumulator[write_m * BLOCK_TILE_N / 4 + write_n];
				//C.incr(g_write_n, g_write_m, s_accumulator[write_m * BLOCK_TILE_N + write_n]);
			}
		}
	}
}


template <typename m_T, int tile_n, int tile_m, int tile_k>
__device__ void d_thread_level_multiply(m_T r_accum[tile_n * tile_m], m_T r_A[tile_m * tile_k], m_T r_B[tile_n * tile_k]) {
	#pragma unroll
	for (int t_k = 0; t_k < tile_k; t_k++) {
		#pragma unroll
		for (int t_m = 0; t_m < tile_m; t_m++)
			#pragma unroll
			for (int t_n = 0; t_n < tile_n; t_n++)
				r_accum[t_m * tile_m + t_n] += r_A[t_m * tile_k + t_k] * r_B[t_k * tile_n + t_n];
	}
}

__global__ void d_cpy_f2_2d(float2 * dst, float * src, int n, int m, int n_new, int m_new) {
	int n_id = threadIdx.x + blockDim.x * blockIdx.x;
	int m_id = threadIdx.y + blockDim.y * blockIdx.y;

	if (m_id < m) {
		if (n_id < n / 2) {
			float *tmp = &src[m_id * n + 2 * n_id];
			dst[m_id * n_new + n_id] = { tmp[0], tmp[1] };
		}
		else if (n_id < n_new) {
			float *tmp = &src[m_id * n + 2 * n_id];
			dst[m_id * n_new + n_id] = { tmp[0], 0 };
		}
		/*if (n_id < n / 2) {
			dst[m_id * n_new + n_id] = reinterpret_cast<float2*>(&src[m_id * n])[n_id];
		}
		else if (n_id < n_new) {
			dst[m_id * n_new + n_id] = { src[m_id * n + n - 1], 0 };
		}
	}
	else if (m_id < m_new) {
		dst[m_id * n_new + n_id] = { 0, 0 };
	}
}*/

/*
__global__ void d_cpy_f4_2d(float4 * dst, float * src, int n, int m, int n_new, int m_new) {
	int n_id = threadIdx.x + blockDim.x * blockIdx.x;
	int m_id = threadIdx.y + blockDim.y * blockIdx.y;

	if (m_id < m) {
		if (n_id < n / 4) {
			float *tmp = &src[m_id * n + 4 * n_id];
			dst[m_id * n_new + n_id] = { tmp[0], tmp[1], tmp[2], tmp[3] };
		}
		else if (n_id < n_new) {
			float *tmp = &src[m_id * n + 4 * n_id];
			switch (n % 4) {
			case 1:
				dst[m_id * n_new + n_id] = { tmp[0], 0, 0, 0 };
				break;
			case 2:
				dst[m_id * n_new + n_id] = { tmp[0], tmp[1], 0, 0 };
				break;
			case 3:
				dst[m_id * n_new + n_id] = { tmp[0], tmp[1], tmp[2], 0 };
				break;
			}
		}
	}
	else if (m_id < m_new) {
		dst[m_id * n_new + n_id] = { 0, 0, 0, 0 };
	}
}
*/
/*
template <typename m_T, mat_order o_T>
void matrix_multiply(d_matrix<m_T, o_T> &A, d_matrix<m_T, o_T> &B, d_matrix<m_T, o_T> &C)
{
	if (A.n_size != B.m_size) {
		throw new exception("Cannot multiply matricies with different K dimensions");
	}

	dim3 threads_per_block(THREAD_SIZE_N, THREAD_SIZE_M);
	dim3 blocks_per_grid(ceil_div(BLOCK_TILE_N, B.n_size), ceil_div(BLOCK_TILE_M, A.m_size));

	d_matrix_multiply<m_T><<<blocks_per_grid, threads_per_block>>>(A, B, C);
}*/

/*void matrix_multiply(float * d_A, float * d_B, float * d_Out, int n_dim, int m_dim, int k_dim)
{
	dim3 threads_per_block(THREAD_SIZE_N, THREAD_SIZE_M);
	dim3 blocks_per_grid(ceil_div(BLOCK_TILE_N, n_dim), ceil_div(BLOCK_TILE_M, m_dim));

	d_matrix_multiply<<<blocks_per_grid, threads_per_block >>>(d_A, d_B, d_Out, n_dim, m_dim, k_dim);
}*/

/*template <mat_order o_T = mat_order::ROW_MAJOR>
void matrix_multiply<float, o_T>(d_matrix<float, o_T>&, d_matrix<float, o_T>&, d_matrix<float, o_T>&);

template <mat_order o_T = mat_order::ROW_MAJOR>
void matrix_multiply<float2, o_T>(d_matrix<float2, o_T> &A, d_matrix<float2, o_T> &B, d_matrix<float2, o_T> &C) {
	if (A.n_size != ceil_div(2, B.m_size)) {
		throw new exception("Cannot multiply matricies with different K dimensions");
	}

	dim3 threads_per_block(THREAD_SIZE_N, THREAD_SIZE_M);
	dim3 blocks_per_grid(ceil_div(BLOCK_TILE_N / 2, B.n_size), ceil_div(BLOCK_TILE_M, A.m_size));

	d_matrix_multiply<float2><<<blocks_per_grid, threads_per_block>>>(A, B, C);
}

template <mat_order o_T = mat_order::ROW_MAJOR>
void matrix_multiply<float4, o_T>(d_matrix<float4, o_T> &A, d_matrix<float4, o_T> &B, d_matrix<float4, o_T> &C) {
	if (A.n_size != ceil_div(4, B.m_size)) {
		throw new exception("Cannot multiply matricies with different K dimensions");
	}

	dim3 threads_per_block(THREAD_SIZE_N, THREAD_SIZE_M);
	dim3 blocks_per_grid(ceil_div(BLOCK_TILE_N / 4, B.n_size), ceil_div(BLOCK_TILE_M, A.m_size));

	d_matrix_multiply<float4><<<blocks_per_grid, threads_per_block>>>(A, B, C);
}*/
/*
template <mat_order o_T>
void create_f1_matrix(d_matrix<float1, o_T> ** matrix, int n_size, int m_size, float * data) {
	cuda_safe_call(cudaMallocManaged(matrix, sizeof(d_matrix<float1>)));

	d_matrix<float1> * h_mat = new d_matrix<float1>();

	h_mat->n_size = n_size;
	h_mat->m_size = m_size;

	cuda_safe_call(cudaMallocManaged(&h_mat->d_data, sizeof(float1) * n_size * m_size));

	//cuda_safe_call(cudaMemcpy(h_mat->d_data, data, sizeof(float) * n_size * m_size, cudaMemcpyHostToDevice));
	h_mat->d_data = reinterpret_cast<float1*>(data);

	cuda_safe_call(cudaMemcpy(*matrix, h_mat, sizeof(d_matrix<float1>), cudaMemcpyHostToDevice));

	free(h_mat);
}

template <mat_order o_T>
void create_f2_matrix(d_matrix<float2, o_T> ** matrix, int n_size, int m_size, float * data) {
	cuda_safe_call(cudaMallocManaged(matrix, sizeof(d_matrix<float2>)));

	d_matrix<float2> * h_mat = new d_matrix<float2>();

	h_mat->n_size = ceil_div(2, n_size);
	h_mat->m_size = 2 * ceil_div(2, m_size);

	cuda_safe_call(cudaMallocManaged(&h_mat->d_data, sizeof(float2) * h_mat->n_size * h_mat->m_size));

	//cuda_safe_call(cudaMemcpy(h_mat->d_data, data, sizeof(float) * n_size * m_size, cudaMemcpyHostToDevice));
	cpy_f2_2d(h_mat->d_data, data, n_size, m_size);

	cuda_safe_call(cudaMemcpy(*matrix, h_mat, sizeof(d_matrix<float2>), cudaMemcpyHostToDevice));

	free(h_mat);
}

template <mat_order o_T>
void create_f4_matrix(d_matrix<float4, o_T> ** matrix, int n_size, int m_size, float * data) {
	cuda_safe_call(cudaMallocManaged(matrix, sizeof(d_matrix<float4>)));

	d_matrix<float4> * h_mat = new d_matrix<float4>();

	h_mat->n_size = ceil_div(4, n_size);
	h_mat->m_size = 4 * ceil_div(4, m_size);

	cuda_safe_call(cudaMallocManaged(&h_mat->d_data, sizeof(float4) * h_mat->n_size * h_mat->m_size));

	//cuda_safe_call(cudaMemcpy(h_mat->d_data, data, sizeof(float) * n_size * m_size, cudaMemcpyHostToDevice));
	cpy_f4_2d(h_mat->d_data, data, n_size, m_size);

	cuda_safe_call(cudaMemcpy(*matrix, h_mat, sizeof(d_matrix<float4>), cudaMemcpyHostToDevice));

	free(h_mat);
}

void cpy_f2_2d(float2 * dst, float * src, int n, int m)
{
	int n_new = ceil_div(2, n);
	int m_new = 2 * ceil_div(2, m);
	dim3 threads_per_block(n_new, m_new);
	dim3 blocks_per_grid(1, 1);
	if (n_new * m > 1024) {
		threads_per_block = dim3(32, 32);
		blocks_per_grid = dim3(ceil_div(32, n_new), ceil_div(32, m_new));
	}

	d_cpy_f2_2d<<<blocks_per_grid, threads_per_block>>>(dst, src, n, m, n_new, m_new);
}

void cpy_f4_2d(float4 * dst, float * src, int n, int m)
{
	int n_new = ceil_div(4, n);
	int m_new = 4 * ceil_div(4, m);
	dim3 threads_per_block(n_new, m_new);
	dim3 blocks_per_grid(1, 1);
	if (n_new * m > 1024) {
		threads_per_block = dim3(32, 32);
		blocks_per_grid = dim3(ceil_div(32, n_new), ceil_div(32, m_new));
	}

	d_cpy_f4_2d<<<blocks_per_grid, threads_per_block>>>(dst, src, n, m, n_new, m_new);
}
*/