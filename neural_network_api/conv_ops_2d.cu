#include "conv_ops_2d.h"
/*
__global__ void d_filter_convolve_2d(
	float * input, 
	float * filter, 
	float * output, 
	shape input_shape, 
	shape filter_shape, 
	shape output_shape, 
	int load_depth, 
	int n_filters,
	int filter_no,
	size_t batch_size) {
	__shared__ float s_filter[MAX_FILTER_SIZE * MAX_FILTER_SIZE * MAX_FILTER_DEPTH];
	__shared__ float s_block[(CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE) * (CONV_BLOCK_SIZE_M + MAX_FILTER_SIZE) * MAX_FILTER_DEPTH];
	//__shared__ float s_conv[CONV_BLOCK_SIZE_N * CONV_BLOCK_SIZE_M * MAX_FILTER_DEPTH];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;
	int k_id = threadIdx.z;

	int tile_n = blockIdx.x * CONV_BLOCK_SIZE_N;
	int tile_m = blockIdx.y * CONV_BLOCK_SIZE_N;
	int f_tile_k = blockIdx.z * CONV_BLOCK_SIZE_N;
	int tile_out_k = blockIdx.z * load_depth;
	int tile_k = tile_out_k * input_shape.depth;

	//fill s_conv with 0s

	/*#pragma unroll
	for (int set_id = 0; set_id < CONV_BLOCK_SIZE_N * CONV_BLOCK_SIZE_M * MAX_FILTER_DEPTH; set_id += CONV_THREAD_SIZE_N * CONV_THREAD_SIZE_M * CONV_THREAD_SIZE_K) {
		s_conv[set_id + k_id * CONV_THREAD_SIZE_N * CONV_THREAD_SIZE_M + m_id * CONV_THREAD_SIZE_N + n_id] = 0;
		//s_conv[0] = 0;
	}

	//load filter (one at a time, for many filters invoke kernel)

	#pragma unroll
	for (int f_load_k = 0; f_load_k < MAX_FILTER_DEPTH; f_load_k += CONV_THREAD_SIZE_K) {
		#pragma unroll
		for (int f_load_m = 0; f_load_m < MAX_FILTER_SIZE; f_load_m += CONV_THREAD_SIZE_M) {
			#pragma unroll
			for (int f_load_n = 0; f_load_n < MAX_FILTER_SIZE; f_load_n += CONV_THREAD_SIZE_N) {
				int load_n_id = f_load_n + n_id;
				int load_m_id = f_load_m + m_id;
				int load_k_id = f_load_k + k_id;

				if (load_n_id < MAX_FILTER_SIZE && load_m_id < MAX_FILTER_SIZE && load_k_id < MAX_FILTER_DEPTH) {
					int s_index = load_k_id * MAX_FILTER_SIZE * MAX_FILTER_SIZE + load_m_id * MAX_FILTER_SIZE + load_n_id;
					int g_index = load_k_id * filter_shape.width * filter_shape.height + load_m_id * filter_shape.width + load_n_id;

					if (load_n_id < filter_shape.width && load_m_id < filter_shape.height && load_k_id < filter_shape.depth)
						s_filter[s_index] = filter[g_index];
					else
						s_filter[s_index] = 0;
				}
			}
		}
	}

	//load block + overlap

	#pragma unroll
	for (int load_k = 0; load_k < MAX_FILTER_DEPTH; load_k += CONV_THREAD_SIZE_K) {
		int load_k_id = load_k + k_id;

		int g_load_k_id = (tile_k + load_k_id) % input_shape.depth;
		int g_load_elem_id = (tile_k + load_k_id) / input_shape.depth;
		int g_in_elem_id = g_load_elem_id * input_shape.width * input_shape.height * input_shape.depth;

		#pragma unroll
		for (int load_m = 0; load_m < CONV_BLOCK_SIZE_M + MAX_FILTER_SIZE; load_m += CONV_THREAD_SIZE_M) {
			#pragma unroll
			for (int load_n = 0; load_n < CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE; load_n += CONV_THREAD_SIZE_N) {
				int load_n_id = load_n + n_id;
				int load_m_id = load_m + m_id;

				if (load_n_id < CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE && load_m_id < CONV_BLOCK_SIZE_M + MAX_FILTER_SIZE && load_k_id < MAX_FILTER_DEPTH) {
					int g_load_n_id = tile_n + load_n_id;
					int g_load_m_id = tile_m + load_m_id;

					int s_index = load_k_id * (CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE) * (CONV_BLOCK_SIZE_M + MAX_FILTER_SIZE) + load_m_id * (CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE) + load_n_id;
					int g_index = g_in_elem_id + g_load_k_id * input_shape.width * input_shape.height + g_load_m_id * input_shape.width + g_load_n_id;

					if (g_load_n_id < input_shape.width && g_load_m_id < input_shape.height && g_load_elem_id < batch_size && load_k_id < load_depth * input_shape.depth) {
						s_block[s_index] = input[g_index];
					}
					else {
						s_block[s_index] = 0;
					}
				}
			}
		}
	}

	cuda_syncthreads();

	//convolve block and compute sums
	for (int elem_k = 0; elem_k < load_depth; elem_k += CONV_THREAD_SIZE_K) {
		int filter_k = elem_k + k_id;
		if (filter_k < load_depth) {
			int filter_k_id = filter_k * filter_shape.depth;

			int out_elem_k = (tile_out_k + filter_k) * output_shape.width * output_shape.height * n_filters + filter_no * output_shape.width * output_shape.height;

			#pragma unroll
			for (int stride_m = 0; stride_m < CONV_THREAD_BLOCK_M; stride_m++) {
				#pragma unroll
				for (int stride_n = 0; stride_n < CONV_THREAD_BLOCK_N; stride_n++) {
					//perform one dot product
					int start_n = n_id * CONV_THREAD_BLOCK_N + stride_n;
					int start_m = m_id * CONV_THREAD_BLOCK_M + stride_m;
					float tmp_sum = 0;
					for (int dot_m = 0; dot_m < filter_shape.height; dot_m++) {
						for (int dot_n = 0; dot_n < filter_shape.width; dot_n++) {
							for (int dot_k = 0; dot_k < filter_shape.depth; dot_k++) {
								int block_elem_k = (filter_k_id + dot_k) * (CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE) * (CONV_BLOCK_SIZE_M + MAX_FILTER_SIZE);

								int block_id = block_elem_k + (start_m + dot_m) * (CONV_BLOCK_SIZE_N + MAX_FILTER_SIZE) + start_n + dot_n;
								int filter_id = dot_k * MAX_FILTER_SIZE * MAX_FILTER_SIZE + dot_m * MAX_FILTER_SIZE + dot_n;
								tmp_sum += s_block[block_id] * s_filter[filter_id];
							}
						}
					}

					int out_n_id = tile_n + start_n;
					int out_m_id = tile_m + start_m;

					if (tile_out_k + filter_k < batch_size && out_n_id < output_shape.width && out_m_id < output_shape.height)
						output[out_elem_k + out_m_id * output_shape.width + out_n_id] = tmp_sum;
				}
			}
		}
	}
}
*/

__global__ void d_filter_convolve_2d(
	float * input,
	float * filter,
	float * output,
	shape input_shape,
	shape filter_shape,
	shape filter_chunks,
	shape output_shape,
	shape padding,
	int filter_no,
	int n_filters,
	size_t batch_size) {

	__shared__ float s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * FILTER_BLOCK_SIZE_K];
	__shared__ float s_load_block[(CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K * CONV_BATCH_DEPTH];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;
	int k_id = threadIdx.z;

	int f_tile_n = blockIdx.x % filter_chunks.width * FILTER_BLOCK_SIZE_N;
	int f_tile_m = blockIdx.y % filter_chunks.height * FILTER_BLOCK_SIZE_M;
	int f_tile_k = blockIdx.z % filter_chunks.depth * FILTER_BLOCK_SIZE_K;

	int b_tile_n = (blockIdx.x / filter_chunks.width) * CONV_BLOCK_SIZE_N;
	int b_tile_m = (blockIdx.y / filter_chunks.height) * CONV_BLOCK_SIZE_M;
	int b_tile_k = f_tile_k;

	int b_elem = (blockIdx.z / filter_chunks.depth) * CONV_BATCH_DEPTH;

	//int b_elem 

	//load filter chunk

	#pragma unroll
	for (int f_load_k = 0; f_load_k < FILTER_BLOCK_SIZE_K; f_load_k += CONV_THREAD_SIZE_K) {
		int load_k_id = f_load_k + k_id;
		int g_load_k_id = f_tile_k + load_k_id;

		#pragma unroll
		for (int f_load_m = 0; f_load_m < FILTER_BLOCK_SIZE_M; f_load_m += CONV_THREAD_SIZE_M) {
			int load_m_id = f_load_m + m_id;
			int g_load_m_id = f_tile_m + load_m_id;

			#pragma unroll
			for (int f_load_n = 0; f_load_n < FILTER_BLOCK_SIZE_N; f_load_n += CONV_THREAD_SIZE_N) {
				int load_n_id = f_load_n + n_id;
				int g_load_n_id = f_tile_n + load_n_id;

				int s_index = load_k_id * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M +
					load_m_id * FILTER_BLOCK_SIZE_N +
					load_n_id;

				if (g_load_n_id < filter_shape.width &&
					g_load_m_id < filter_shape.height &&
					g_load_k_id < filter_shape.depth &&
					b_elem < batch_size) {

					int g_index = g_load_k_id * filter_shape.width * filter_shape.height +
						g_load_m_id * filter_shape.width +
						g_load_n_id;

					s_filter[s_index] = filter[g_index];
				}
				else {
					s_filter[s_index] = 0.0;
				}
			}
		}
	}

	cuda_syncthreads();

	//load input chunk

	#pragma unroll
	for (int batch = 0; batch < CONV_BATCH_DEPTH; batch++) {
		#pragma unroll
		for (int b_load_k = 0; b_load_k < CONV_BLOCK_SIZE_K; b_load_k += CONV_THREAD_SIZE_K) {
			int load_k_id = b_load_k + k_id;
			int g_load_k_id = b_tile_k + load_k_id;

			#pragma unroll
			for (int b_load_m = 0; b_load_m < CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M; b_load_m += CONV_THREAD_SIZE_M) {
				int load_m_id = b_load_m + m_id;
				int g_load_m_id = f_tile_m + b_tile_m + load_m_id - padding.height;

				#pragma unroll
				for (int b_load_n = 0; b_load_n < CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N; b_load_n += CONV_THREAD_SIZE_N) {
					int load_n_id = b_load_n + n_id;
					int g_load_n_id = f_tile_n + b_tile_n + load_n_id - padding.width;

					int s_index = batch * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K +
						load_k_id * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) +
						load_m_id * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) +
						load_n_id;

					if (0 <= g_load_n_id &&
						g_load_n_id < input_shape.width &&
						0 <= g_load_m_id &&
						g_load_m_id < input_shape.height &&
						g_load_k_id < input_shape.depth &&
						b_elem + batch < batch_size) {

						int g_index = (b_elem + batch) * input_shape.width * input_shape.height * input_shape.depth +
							g_load_k_id * input_shape.width * input_shape.height +
							g_load_m_id * input_shape.width +
							g_load_n_id;

						s_load_block[s_index] = input[g_index];
					}
					else {
						s_load_block[s_index] = 0.0;
					}
				}
			}
		}
	}

	cuda_syncthreads();

	//convolve and write

	#pragma unroll
	for (int batch = 0; batch < CONV_BATCH_DEPTH; batch++) {
		if (k_id == 0) {
			#pragma unroll
			for (int stride_m = 0; stride_m < CONV_THREAD_BLOCK_M; stride_m++) {
				int start_m = m_id * CONV_THREAD_BLOCK_M + stride_m;
				#pragma unroll
				for (int stride_n = 0; stride_n < CONV_THREAD_BLOCK_N; stride_n++) {
					int start_n = n_id * CONV_THREAD_BLOCK_N + stride_n;

					int out_n_id = b_tile_n + start_n;
					int out_m_id = b_tile_m + start_m;
					int out_k_id = (b_elem + batch) * n_filters + filter_no;

					if (out_n_id < output_shape.width &&
						out_m_id < output_shape.height &&
						b_elem + batch < batch_size) {

						int out_index = out_k_id * output_shape.width * output_shape.height +
							out_m_id * output_shape.width +
							out_n_id;

						float inc = calculate_conv2d_dot<CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N, CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M, CONV_THREAD_BLOCK_K>(
							s_filter,
							&s_load_block[batch * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K],
							start_n,
							start_m
							);

						atomic_add(&output[out_index], inc);
					}
				}
			}
		}
	}

	cuda_syncthreads();
}

/*
__global__ void d_filter_outer_convolve_2d(
	const float * input, 
	const float * filter, 
	volatile float * output, 
	const shape input_shape, 
	const shape filter_shape, 
	const shape output_shape, 
	const int filter_no) {
	__shared__ float s_filter[MAX_FILTER_SIZE * MAX_FILTER_SIZE * MAX_FILTER_DEPTH];
	__shared__ float s_load_block[(CONV_EXP_BLOCK_SIZE_N + MAX_FILTER_SIZE) * (CONV_EXP_BLOCK_SIZE_M + MAX_FILTER_SIZE)];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;

	int tile_n = blockIdx.x * CONV_EXP_BLOCK_SIZE_N;
	int tile_m = blockIdx.y * CONV_EXP_BLOCK_SIZE_M;

	int k_id = blockIdx.z;

	//load filter into shared memory
	for (int f_load_k = 0; f_load_k < filter_shape.depth; f_load_k++) {
		#pragma unroll
		for (int f_load_m = 0; f_load_m < MAX_FILTER_SIZE; f_load_m += CONV_EXP_THREAD_SIZE_M) {
			#pragma unroll
			for (int f_load_n = 0; f_load_n < MAX_FILTER_SIZE; f_load_n += CONV_EXP_THREAD_SIZE_N) {
				int load_n = f_load_n + n_id;
				int load_m = f_load_m + m_id;

				int s_index = f_load_k * MAX_FILTER_SIZE * MAX_FILTER_SIZE + load_m * MAX_FILTER_SIZE + load_n;

				if (load_n < filter_shape.width && load_m < filter_shape.height) {
					s_filter[s_index] = filter[f_load_k * filter_shape.width * filter_shape.height + load_m * filter_shape.width + load_n];
				}
				else {
					s_filter[s_index] = 0;
				}
			}
		}
	}

	cuda_syncthreads();

	//load block with overlap into shared memory
	#pragma unroll
	for (int in_m = 0; in_m < CONV_EXP_BLOCK_SIZE_M + MAX_FILTER_SIZE; in_m += CONV_EXP_THREAD_SIZE_M) {
		#pragma unroll
		for (int in_n = 0; in_n < CONV_EXP_BLOCK_SIZE_N + MAX_FILTER_SIZE; in_n += CONV_EXP_THREAD_SIZE_N) {
			int start_n = in_n + n_id;
			int start_m = in_m + m_id;

			int g_in_n = tile_n + start_n - MAX_FILTER_SIZE / 2;
			int g_in_m = tile_m + start_m - MAX_FILTER_SIZE / 2;

			int s_index = start_m * (CONV_EXP_BLOCK_SIZE_N + MAX_FILTER_SIZE) + start_n;

			if (0 <= g_in_n && g_in_n < input_shape.width && 0 <= g_in_m && g_in_m < input_shape.height) {
				//int out_index = filter_k * (CONV_EXP_BLOCK_SIZE_N + MAX_FILTER_SIZE) * (CONV_EXP_BLOCK_SIZE_M + MAX_FILTER_SIZE) + (start_m + filter_m) * (CONV_EXP_BLOCK_SIZE_N + MAX_FILTER_SIZE) + start_n + filter_n;
				int in_index = k_id * input_shape.width * input_shape.height * input_shape.depth + filter_no * input_shape.width * input_shape.height + g_in_m * input_shape.width + g_in_n;
				float load_elem = input[in_index];
				s_load_block[s_index] = input[in_index];
			}
			else {
				s_load_block[s_index] = 0;
			}
		}
	}

	cuda_syncthreads();

	//accumulate to global memory
	
	#pragma unroll
	for (int out_m = 0; out_m < CONV_EXP_BLOCK_SIZE_M; out_m += CONV_EXP_THREAD_SIZE_M) {
		#pragma unroll
		for (int out_n = 0; out_n < CONV_EXP_BLOCK_SIZE_N; out_n += CONV_EXP_THREAD_SIZE_N) {
			int start_n_id = out_n + n_id;
			int start_m_id = out_m + m_id;

			for (int filter_k = 0; filter_k < filter_shape.depth; filter_k++) {
				float accum = 0;

				for (int filter_m = 0; filter_m < filter_shape.height; filter_m++) {
					for (int filter_n = 0; filter_n < filter_shape.width; filter_n++) {
						accum += s_load_block[(start_m_id + filter_m) * (CONV_EXP_BLOCK_SIZE_N + MAX_FILTER_SIZE) + start_n_id + filter_n] *
							s_filter[filter_k * MAX_FILTER_SIZE * MAX_FILTER_SIZE + filter_m * MAX_FILTER_SIZE + filter_n];
					}
				}

				int g_out_n = tile_n + start_n_id - MAX_FILTER_SIZE / 2 + filter_shape.width - 1;
				int g_out_m = tile_m + start_m_id - MAX_FILTER_SIZE / 2 + filter_shape.height - 1;
				int g_out_elem = k_id * output_shape.width * output_shape.height * output_shape.depth;
				
				if (0 <= g_out_n && g_out_n < output_shape.width && 0 <= g_out_m && g_out_m < output_shape.height) {
					int out_index = g_out_elem + filter_k * output_shape.width * output_shape.height + g_out_m * output_shape.width + g_out_n;
					output[out_index] += accum;
				}
					
			}
		}
	}
}
*/

__global__ void d_filter_outer_convolve_2d(
	float * input,
	float * filter,
	float * output,
	shape input_shape,
	shape filter_shape,
	shape filter_chunks,
	shape output_shape,
	shape padding,
	int input_depth,
	int filter_no,
	size_t batch_size) {

	__shared__ float s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * FILTER_BLOCK_SIZE_K];
	__shared__ float s_load_block[(CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_OUTER_BLOCK_SIZE_K];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;
	int k_id = threadIdx.z;

	int f_tile_n = blockIdx.x % filter_chunks.width;
	int f_tile_m = blockIdx.y % filter_chunks.height;
	int f_tile_k = blockIdx.z % filter_chunks.depth;

	int b_tile_n = blockIdx.x / filter_chunks.width * CONV_OUTER_BLOCK_SIZE_N;
	int b_tile_m = blockIdx.y / filter_chunks.height * CONV_OUTER_BLOCK_SIZE_M;
	int b_tile_k = (blockIdx.z / filter_chunks.depth) % input_depth * CONV_OUTER_BLOCK_SIZE_K;

	int b_elem = blockIdx.z / (filter_chunks.depth * input_depth);

	//load filter chunk

	#pragma unroll
	for (int f_load_k = 0; f_load_k < FILTER_BLOCK_SIZE_K; f_load_k += CONV_OUTER_THREAD_SIZE_K) {
		int load_k_id = f_load_k + k_id;
		int g_load_k_id = f_tile_k + load_k_id;

		#pragma unroll
		for (int f_load_m = 0; f_load_m < FILTER_BLOCK_SIZE_M; f_load_m += CONV_OUTER_THREAD_SIZE_M) {
			int load_m_id = f_load_m + m_id;
			int g_load_m_id = f_tile_m + load_m_id;

			#pragma unroll
			for (int f_load_n = 0; f_load_n < FILTER_BLOCK_SIZE_N; f_load_n += CONV_OUTER_THREAD_SIZE_N) {
				int load_n_id = f_load_n + n_id;
				int g_load_n_id = f_tile_n + load_n_id;
				
				int s_index = load_k_id * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M +
					load_m_id * FILTER_BLOCK_SIZE_N +
					load_n_id;

				if (g_load_n_id < filter_shape.width &&
					g_load_m_id < filter_shape.height &&
					g_load_k_id < filter_shape.depth) {

					int g_index = g_load_k_id * filter_shape.width * filter_shape.height +
						g_load_m_id * filter_shape.width +
						g_load_n_id;

					s_filter[s_index] = filter[g_index];
				}
				else {
					s_filter[s_index] = 0.0;
				}
			}
		}
	}

	cuda_syncthreads();

	//load input block + edge

	#pragma unroll
	for (int b_load_k = 0; b_load_k < CONV_OUTER_BLOCK_SIZE_K; b_load_k += CONV_OUTER_THREAD_SIZE_K) {
		int load_k_id = b_load_k + k_id;
		int g_load_k_id = b_tile_k + load_k_id;

		#pragma unroll
		for (int b_load_m = 0; b_load_m < CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M; b_load_m += CONV_OUTER_THREAD_SIZE_M) {
			int load_m_id = b_load_m + m_id;
			int g_load_m_id = b_tile_m + load_m_id - filter_shape.height + f_tile_m + 1 + padding.height;

			#pragma unroll
			for (int b_load_n = 0; b_load_n < CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N; b_load_n += CONV_OUTER_THREAD_SIZE_N) {
				int load_n_id = b_load_n + n_id;
				int g_load_n_id = b_tile_n + load_n_id - filter_shape.width + f_tile_n + 1 + padding.width;

				int s_index = load_k_id * (CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) +
					load_m_id * (CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) +
					load_n_id;

				if (0 <= g_load_n_id &&
					g_load_n_id < input_shape.width &&
					0 <= g_load_m_id &&
					g_load_m_id < input_shape.height &&
					filter_no + g_load_k_id < input_shape.depth &&
					b_elem < batch_size) {

					int g_index = b_elem * input_shape.width * input_shape.height * input_shape.depth +
						(filter_no + g_load_k_id) * input_shape.width * input_shape.height +
						g_load_m_id * input_shape.width +
						g_load_n_id;

					s_load_block[s_index] = input[g_index];
				}
				else {
					s_load_block[s_index] = 0.0;
				}
			}
		}
	}

	cuda_syncthreads();

	//convole and accumulate to output

	#pragma unroll
	for (int f_k = 0; f_k < FILTER_BLOCK_SIZE_K; f_k++) {
		#pragma unroll
		for (int b_k = 0; b_k < CONV_OUTER_THREAD_BLOCK_K; b_k++) {
			int b_k_id = k_id * CONV_OUTER_THREAD_BLOCK_K + b_k;
			int out_k_id = (b_tile_k + b_k_id) * output_shape.depth + f_k;

			#pragma unroll
			for (int b_m = 0; b_m < CONV_OUTER_THREAD_BLOCK_M; b_m++) {
				int b_m_id = m_id * CONV_OUTER_THREAD_BLOCK_M + b_m;
				int out_m_id = b_tile_m + b_m_id;

				#pragma unroll
				for (int b_n = 0; b_n < CONV_OUTER_THREAD_BLOCK_N; b_n++) {
					int b_n_id = n_id * CONV_OUTER_THREAD_BLOCK_N + b_n;
					int out_n_id = b_tile_n + b_n_id;

					if (out_n_id < output_shape.width &&
						out_m_id < output_shape.height &&
						out_k_id < output_shape.depth &&
						b_elem < batch_size) {

						int out_index = (b_elem * output_shape.depth + out_k_id) * output_shape.width * output_shape.height +
							out_m_id * output_shape.width +
							out_n_id;

						float inc = calculate_conv2d_dot<CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N, CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M, CONV_OUTER_THREAD_BLOCK_K>(
							s_filter,
							s_load_block,
							b_n_id,
							b_m_id
						);

						atomic_add(&output[out_index], inc);
					}
				}
			}
		}
	}
}

__global__ void d_filter_convolve_2d_derivative(
	float * input,
	float * filter,
	float * output,
	shape input_shape,
	shape filter_shape,
	shape filter_chunks,
	shape output_shape,
	shape padding,
	int input_depth,
	size_t batch_size) {

	__shared__ float s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * FILTER_BLOCK_SIZE_K];
	__shared__ float s_load_block[(CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;
	int k_id = threadIdx.z;

	int f_tile_n = blockIdx.x % filter_chunks.width * FILTER_BLOCK_SIZE_N;
	int f_tile_m = blockIdx.y % filter_chunks.height * FILTER_BLOCK_SIZE_M;
	int f_tile_k = blockIdx.z % filter_chunks.depth * FILTER_BLOCK_SIZE_K;

	int b_tile_n = (blockIdx.x / filter_chunks.width) * CONV_DER_BLOCK_SIZE_N;
	int b_tile_m = (blockIdx.y / filter_chunks.height) * CONV_DER_BLOCK_SIZE_M;
	int b_tile_k = (blockIdx.z / filter_chunks.depth) % input_depth * CONV_DER_BLOCK_SIZE_K;

	int b_elem = blockIdx.z / (filter_chunks.depth * input_depth);

	//int b_elem 

	//load filter chunk

	#pragma unroll
	for (int f_load_k = 0; f_load_k < FILTER_BLOCK_SIZE_K; f_load_k += CONV_DER_THREAD_SIZE_K) {
		int load_k_id = f_load_k + k_id;
		int g_load_k_id = f_tile_k + load_k_id;

		#pragma unroll
		for (int f_load_m = 0; f_load_m < FILTER_BLOCK_SIZE_M; f_load_m += CONV_DER_THREAD_SIZE_M) {
			int load_m_id = f_load_m + m_id;
			int g_load_m_id = f_tile_m + load_m_id;

			#pragma unroll
			for (int f_load_n = 0; f_load_n < FILTER_BLOCK_SIZE_N; f_load_n += CONV_DER_THREAD_SIZE_N) {
				int load_n_id = f_load_n + n_id;
				int g_load_n_id = f_tile_n + load_n_id;

				int s_index = load_k_id * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M +
					load_m_id * FILTER_BLOCK_SIZE_N +
					load_n_id;

				if (g_load_n_id < filter_shape.width &&
					g_load_m_id < filter_shape.height &&
					g_load_k_id < filter_shape.depth) {

					int g_index = b_elem * filter_shape.width * filter_shape.height * filter_shape.depth +
						g_load_k_id * filter_shape.width * filter_shape.height +
						g_load_m_id * filter_shape.width +
						g_load_n_id;

					s_filter[s_index] = filter[g_index];
				}
				else {
					s_filter[s_index] = 0.0;
				}
			}
		}
	}

	cuda_syncthreads();

	//load input chunk

	#pragma unroll
	for (int b_load_k = 0; b_load_k < CONV_DER_BLOCK_SIZE_K; b_load_k += CONV_DER_THREAD_SIZE_K) {
		int load_k_id = b_load_k + k_id;
		int g_load_k_id = b_tile_k + load_k_id;

		#pragma unroll
		for (int b_load_m = 0; b_load_m < CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M; b_load_m += CONV_DER_THREAD_SIZE_M) {
			int load_m_id = b_load_m + m_id;
			int g_load_m_id = f_tile_m + b_tile_m + load_m_id - padding.height;

			#pragma unroll
			for (int b_load_n = 0; b_load_n < CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N; b_load_n += CONV_DER_THREAD_SIZE_N) {
				int load_n_id = b_load_n + n_id;
				int g_load_n_id = f_tile_n + b_tile_n + load_n_id - padding.width;

				int s_index = load_k_id * (CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) +
					load_m_id * (CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) +
					load_n_id;

				if (0 <= g_load_n_id &&
					g_load_n_id < input_shape.width &&
					0 <= g_load_m_id &&
					g_load_m_id < input_shape.height &&
					g_load_k_id < input_shape.depth) {

					int g_index = b_elem * input_shape.width * input_shape.height * input_shape.depth +
						g_load_k_id * input_shape.width * input_shape.height +
						g_load_m_id * input_shape.width +
						g_load_n_id;

					s_load_block[s_index] = input[g_index];
				}
				else {
					s_load_block[s_index] = 0.0;
				}
			}
		}
	}

	cuda_syncthreads();

	//convolve and write

	#pragma unroll
	for (int filter_k = 0; filter_k < FILTER_BLOCK_SIZE_K; filter_k ++) {
		#pragma unroll
		for (int stride_k = 0; stride_k < CONV_DER_THREAD_BLOCK_K; stride_k++) {
			int start_k = k_id * CONV_DER_THREAD_BLOCK_K + stride_k;
			#pragma unroll
			for (int stride_m = 0; stride_m < CONV_DER_THREAD_BLOCK_M; stride_m++) {
				int start_m = m_id * CONV_DER_THREAD_BLOCK_M + stride_m;
				#pragma unroll
				for (int stride_n = 0; stride_n < CONV_DER_THREAD_BLOCK_N; stride_n++) {
					int start_n = n_id * CONV_DER_THREAD_BLOCK_N + stride_n;

					int out_n_id = b_tile_n + start_n;
					int out_m_id = b_tile_m + start_m;
					int out_layer_id = b_tile_k + start_k;
					int out_filter_id = f_tile_k + filter_k;
					int out_k_id = out_filter_id * output_shape.depth + out_layer_id;

					if (out_n_id < output_shape.width &&
						out_m_id < output_shape.height &&
						out_filter_id < filter_shape.depth &&
						out_layer_id < output_shape.depth) {

						int out_index = out_k_id * output_shape.width * output_shape.height +
							out_m_id * output_shape.width +
							out_n_id;

						float inc = calculate_conv2d_dot<CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N, CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M, 1>(
							&s_filter[filter_k * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M],
							&s_load_block[start_k * (CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M)],
							start_n,
							start_m
						);

						atomic_add(&output[out_index], inc);
					}
				}
			}
		}
	}

	cuda_syncthreads();
}

/*
template <unsigned int block_size>
__global__ void d_filter_convolve_2d_derivative(
	float * input,
	float * pds,
	float * output,
	shape input_shape,
	shape pd_shape,
	shape output_shape,
	int filter_no) {

	__shared__ float s_load[block_size * block_size * MAX_FILTER_DEPTH];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;
	int k_id = threadIdx.z;

	int t_id = k_id * CONV_DER_THREAD_SIZE_N * CONV_DER_THREAD_SIZE_M + m_id * CONV_DER_THREAD_SIZE_N + n_id;

	int filter_n = blockIdx.x;
	int filter_m = blockIdx.y;

	int elem = blockIdx.z;

	//iterate through per thread for each output element, load required and dot <-- may have to use this (dynamic shared memory load and mul, then parallel reduction)


	//MAYBE expand this so larger images can be used
	//load an multiply elements

	#pragma unroll
	for (int load_m = 0; load_m < block_size; load_m += CONV_DER_THREAD_SIZE_M) {
		#pragma unroll
		for (int load_n = 0; load_n < block_size; load_n += CONV_DER_THREAD_SIZE_N) {
			int load_n_id = load_n + n_id;
			int load_m_id = load_m + m_id;

			if (load_n_id >= block_size || load_m_id >= block_size)
				continue;

			if (load_n_id < pd_shape.width && load_m_id < pd_shape.height) {
				int pd_index = (elem * pd_shape.depth + filter_no) * pd_shape.width * pd_shape.height + load_m_id * pd_shape.width + load_n_id;
				float pd_val = pds[pd_index];
				#pragma unroll
				for (int load_k = 0; load_k < MAX_FILTER_DEPTH; load_k += CONV_DER_THREAD_SIZE_K) {
					int load_k_id = load_k + k_id;

					int in_index = elem * input_shape.width * input_shape.height * input_shape.depth + load_k_id * input_shape.width * input_shape.height + (load_m_id + filter_m) * input_shape.width + (load_n_id + filter_n);

					int s_index = load_k_id * block_size * block_size + load_m_id * block_size + load_n_id;
					if (load_k_id < input_shape.depth)
						s_load[s_index] = pd_val * input[in_index];
					else
						s_load[s_index] = 0;
				}
			}
			else {
				#pragma unroll
				for (int load_k = 0; load_k < MAX_FILTER_DEPTH; load_k += CONV_DER_THREAD_SIZE_K) {
					int load_k_id = load_k + k_id;
					int s_index = load_k_id * block_size * block_size + load_m_id * block_size + load_n_id;
					s_load[s_index] = 0;
				}
			}
		}
	}

	cuda_syncthreads();

	//parallel reduction in each layer

	#pragma unroll
	for (int layer = 0; layer < block_size * block_size * MAX_FILTER_DEPTH; layer += block_size * block_size) {
		#pragma unroll
		for (int reduce = 0; reduce < 1024; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 2048) {
				if (t_id < 1024) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 1024];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 512; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 1024) {
				if (t_id < 512) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 512];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 256; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 512) {
				if (t_id < 256) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 256];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 128; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 256) {
				if (t_id < 128) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 128];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 64; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 128) {
				if (t_id < 64) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 64];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 32; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 64) {
				if (t_id < 32) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 32];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 16; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 32) {
				if (t_id < 16) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 16];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 8; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 16) {
				if (t_id < 8) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 8];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 4; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 8) {
				if (t_id < 4) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 4];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 2; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 4) {
				if (t_id < 2) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 2];
				}
				cuda_syncthreads();
			}
		}
		#pragma unroll
		for (int reduce = 0; reduce < 1; reduce += CONV_DER_THREAD_SIZE) {
			if (block_size * block_size >= 2) {
				if (t_id < 1) {
					s_load[layer + reduce + t_id] += s_load[layer + reduce + t_id + 1];
				}
				cuda_syncthreads();
			}
		}
	}

	#pragma unroll
	for (int out_k = 0; out_k < MAX_FILTER_DEPTH; out_k++) {
		if (t_id == 0 && out_k < output_shape.depth) {
			output[out_k * output_shape.width * output_shape.height + filter_m * output_shape.width + filter_n] += s_load[out_k * block_size * block_size];
		}
	}
}
*/

__global__ void d_pool_2d(
	float * input, 
	int * mask, 
	float * output, 
	shape input_shape, 
	shape pool_size, 
	shape stride, 
	shape output_shape, 
	size_t batch_size) {
	__shared__ float s_block[(POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) * (POOL_BLOCK_SIZE_M + MAX_POOL_SIZE) * POOL_BLOCK_DEPTH];

	int n_id = threadIdx.x;
	int m_id = threadIdx.y;
	int k_id = threadIdx.z;

	int n_tile = blockIdx.x * POOL_BLOCK_SIZE_N;
	int m_tile = blockIdx.y * POOL_BLOCK_SIZE_M;
	int k_tile = blockIdx.z * POOL_BLOCK_DEPTH;

	//load image block into s_block
	//each block convolves one region of the image

	#pragma unroll
	for (int load_k = 0; load_k < POOL_BLOCK_DEPTH; load_k += POOL_THREAD_SIZE_K) {
		int load_k_id = load_k + k_id;

		int g_load_k_id = (load_k_id + k_tile) % input_shape.depth;
		int g_load_elem_id = ((load_k_id + k_tile) / input_shape.depth);
		int g_in_elem_id = g_load_elem_id * (input_shape.width * input_shape.height * input_shape.depth);

		int g_out_elem_id = g_load_elem_id * (output_shape.width * output_shape.height * output_shape.depth);

		#pragma unroll
		for (int load_m = 0; load_m < POOL_BLOCK_SIZE_M + MAX_POOL_SIZE; load_m += POOL_THREAD_SIZE_M) {
			#pragma unroll
			for (int load_n = 0; load_n < POOL_BLOCK_SIZE_N + MAX_POOL_SIZE; load_n += POOL_THREAD_SIZE_N) {
				int load_m_id = load_m + m_id;
				int load_n_id = load_n + n_id;

				int g_load_m_id = load_m_id + m_tile;
				int g_load_n_id = load_n_id + n_tile;

				int s_load_id = load_k_id * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) * (POOL_BLOCK_SIZE_M + MAX_POOL_SIZE) + load_m_id * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) + load_n_id;
				//int g_load_id = g_in_elem_id + (g_load_m_id * input_shape.width + g_load_n_id) * input_shape.depth + g_load_k_id;
				int g_load_id = g_in_elem_id + g_load_k_id * input_shape.width * input_shape.height + g_load_m_id * input_shape.width + g_load_n_id;

				if (g_load_n_id < input_shape.width && g_load_m_id < input_shape.height && g_load_elem_id < batch_size) {
					s_block[s_load_id] = input[g_load_id];
				}
				else {
					s_block[s_load_id] = -99999;
				}
			}
		}

		cuda_syncthreads();

		//loop through each stride with even divisions among threads
		//each thread iterates through every pool with the top left corner in the thread's allocated block

		int n_offset = ((input_shape.width - pool_size.width) - (n_tile + n_id * POOL_THREAD_BLOCK_N)) % stride.width;
		int m_offset = ((input_shape.height - pool_size.height) - (m_tile + m_id * POOL_THREAD_BLOCK_M)) % stride.height;

		//int k_offset = ((input_shape.depth - pool_size.depth) - (k_tile _ k_id * POOL_THREAD_BLOCK_K)) % stride.depth;

		for (int stride_m = m_offset; stride_m < POOL_THREAD_BLOCK_M; stride_m += stride.height) {
			for (int stride_n = n_offset; stride_n < POOL_THREAD_BLOCK_N; stride_n += stride.width) {
				int stride_index_n = n_id * POOL_THREAD_BLOCK_N + stride_n;
				int stride_index_m = m_id * POOL_THREAD_BLOCK_M + stride_m;
				float tmp_max = -99999;
				int n_index = stride_index_n;
				int m_index = stride_index_m;
				for (int pool_m = 0; pool_m < pool_size.height; pool_m++) {
					for (int pool_n = 0; pool_n < pool_size.width; pool_n++) {
						//float tmp_read = s_block[((stride_index_m + pool_m) * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) + stride_index_n + pool_n) * POOL_BLOCK_DEPTH + load_k_id];
						float tmp_read = s_block[load_k_id * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) * (POOL_BLOCK_SIZE_M + MAX_POOL_SIZE) +
							(stride_index_m + pool_m) * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) +
							stride_index_n + pool_n];
						if (tmp_read > tmp_max) {
							tmp_max = tmp_read;
							n_index = stride_index_n + pool_n;
							m_index = stride_index_m + pool_m;
						}
					}
				}
				//write tmp_max to output
				int g_out_n = (n_tile + stride_index_n) / stride.width;
				int g_out_m = (m_tile + stride_index_m) / stride.height;

				//int g_mask_out_n = n_tile + n_index;
				//int g_mask_out_m = m_tile + m_index;
				int mask_out_index = (m_tile + m_index) * input_shape.width + (n_tile + n_index);

				if (g_out_n < output_shape.width && g_out_m < output_shape.height && g_load_elem_id < batch_size) {
					//output[g_out_elem_id + (g_out_m * output_shape.width + g_out_n) * output_shape.depth + g_load_k_id] = tmp_max;
					//mask[g_out_elem_id + (g_out_m * output_shape.width + g_out_n) * output_shape.depth + g_load_k_id] = mask_out_index;

					output[g_out_elem_id + g_load_k_id * output_shape.width * output_shape.height + g_out_m * output_shape.width + g_out_n] = tmp_max;
					mask[g_out_elem_id + g_load_k_id * output_shape.width * output_shape.height + g_out_m * output_shape.width + g_out_n] = mask_out_index;

					//mask[g_in_elem_id + (g_mask_out_m * input_shape.width + g_mask_out_n) * input_shape.depth + g_load_k_id] = 1;
				}
			}
		}
	}
}

__global__ void d_pool_2d_derivative(
	float * input, 
	int * mask, 
	float * output, 
	size_t input_size, 
	size_t output_size, 
	size_t batch_size) {
	int t_id = threadIdx.x + blockIdx.x * blockDim.x;
	int batch_index = blockIdx.y;

	if (t_id < input_size && batch_index < batch_size) {
		int out_id = mask[batch_index * input_size + t_id];
		output[batch_index * output_size + out_id] += input[batch_index * input_size + t_id];
	}
}

template <int BLOCK_N, int BLOCK_M, int DEPTH>
__device__ float calculate_conv2d_dot(volatile float * s_filter, volatile float * s_load_block, int start_n, int start_m)
{
	float accum = 0;

	#pragma unroll
	for (int dot_k = 0; dot_k < DEPTH; dot_k++) {
		#pragma unroll
		for (int dot_m = 0; dot_m < FILTER_BLOCK_SIZE_M; dot_m++) {
			#pragma unroll
			for (int dot_n = 0; dot_n < FILTER_BLOCK_SIZE_N; dot_n++) {
				int f_index = dot_k * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M + dot_m * FILTER_BLOCK_SIZE_N + dot_n;
				int b_index = dot_k * BLOCK_N * BLOCK_M + (dot_m + start_m) * BLOCK_N + (dot_n + start_n);

				accum += s_filter[f_index] * s_load_block[b_index];
			}
		}
	}

	return accum;
}

void filter_convolve_2d(
	float * d_input, 
	float * d_filter, 
	float * d_output, 
	shape input_shape, 
	shape output_shape, 
	shape filter_shape, 
	shape padding,
	size_t batch_size)
{
	//dim3 threads_per_block(CONV_THREAD_SIZE_N, CONV_THREAD_SIZE_M, CONV_THREAD_SIZE_K);
	//dim3 blocks_per_grid(ceil_div(CONV_BLOCK_SIZE_N, input_shape.width), ceil_div(CONV_BLOCK_SIZE_M, input_shape.height), ceil_div(MAX_FILTER_DEPTH, input_shape.depth * batch_size));

	shape filter_chunks = shape(
		ceil_div(FILTER_BLOCK_SIZE_N, filter_shape.width), 
		ceil_div(FILTER_BLOCK_SIZE_M, filter_shape.height), 
		ceil_div(FILTER_BLOCK_SIZE_K, filter_shape.depth)
	);

	dim3 threads_per_block(CONV_THREAD_SIZE_N, CONV_THREAD_SIZE_M, CONV_THREAD_SIZE_K);
	dim3 blocks_per_grid(
		ceil_div(CONV_BLOCK_SIZE_N, input_shape.width + padding.width) * filter_chunks.width, 
		ceil_div(CONV_BLOCK_SIZE_M, input_shape.height + padding.width) * filter_chunks.height, 
		ceil_div(CONV_BATCH_DEPTH, batch_size) * filter_chunks.depth
	);

	for (int filter = 0; filter < output_shape.depth; filter++)
	{
		/*d_filter_convolve_2d<<<blocks_per_grid, threads_per_block>>>(
			d_input, 
			&d_filter[filter_shape.size() * filter], 
			d_output, 
			input_shape, 
			filter_shape, 
			output_shape, 
			MAX_FILTER_DEPTH / input_shape.depth, 
			output_shape.depth,
			filter,
			batch_size
		);*/

		d_filter_convolve_2d<<<blocks_per_grid, threads_per_block>>>(
			d_input,
			&d_filter[filter_shape.size() * filter],
			d_output,
			input_shape,
			filter_shape,
			filter_chunks,
			output_shape,
			padding,
			filter,
			output_shape.depth,
			batch_size
		);
	}
}

void pool_2d(
	float * d_input, 
	int * d_mask, 
	float * d_output, 
	shape input_shape, 
	shape pool_size, 
	shape stride, 
	shape output_shape, 
	size_t batch_size)
{
	dim3 threads_per_block(POOL_THREAD_SIZE_N, POOL_THREAD_SIZE_M, POOL_THREAD_SIZE_K);
	dim3 blocks_per_grid(ceil_div(POOL_BLOCK_SIZE_N, input_shape.width), ceil_div(POOL_BLOCK_SIZE_M, input_shape.height), ceil_div(POOL_BLOCK_DEPTH, input_shape.depth * batch_size));
	
	d_pool_2d<<<blocks_per_grid, threads_per_block>>>(d_input, d_mask, d_output, input_shape, pool_size, stride, output_shape, batch_size);
	/*size_t max_pool_dim = pool_size.max_dim();

	if (max_pool_dim <= 2) {
		d_pool_2d<2><<<blocks_per_grid, threads_per_block>>>(d_input, d_mask, d_output, input_shape, pool_size, stride, output_shape, batch_size);
	}
	else if (max_pool_dim <= 4) {
		d_pool_2d<4><<<blocks_per_grid, threads_per_block>>>(d_input, d_mask, d_output, input_shape, pool_size, stride, output_shape, batch_size);
	}
	else if (max_pool_dim <= 8) {
		d_pool_2d<4><<<blocks_per_grid, threads_per_block>>>(d_input, d_mask, d_output, input_shape, pool_size, stride, output_shape, batch_size);
	}
	else {
		throw new exception("Pool size too large");
	}*/
}

void filter_outer_convolve_2d(
	float * d_input, 
	float * d_filter, 
	float * d_output, 
	shape input_shape, 
	shape output_shape, 
	shape filter_shape, 
	shape padding,
	size_t batch_size)
{
	shape filter_chunks = shape(
		ceil_div(FILTER_BLOCK_SIZE_N, filter_shape.width),
		ceil_div(FILTER_BLOCK_SIZE_M, filter_shape.height),
		ceil_div(FILTER_BLOCK_SIZE_K, filter_shape.depth)
	);

	//dim3 threads_per_block(CONV_EXP_THREAD_SIZE_N, CONV_EXP_THREAD_SIZE_M, 1);
	//dim3 blocks_per_grid(ceil_div(CONV_EXP_BLOCK_SIZE_N, input_shape.width), ceil_div(CONV_EXP_BLOCK_SIZE_M, input_shape.height), batch_size);
	dim3 threads_per_block(CONV_OUTER_THREAD_SIZE_N, CONV_OUTER_THREAD_SIZE_M, CONV_OUTER_THREAD_SIZE_K);
	dim3 blocks_per_grid(
		ceil_div(CONV_OUTER_BLOCK_SIZE_N, input_shape.width) * filter_chunks.width,
		ceil_div(CONV_OUTER_BLOCK_SIZE_M, input_shape.height) * filter_chunks.height,
		ceil_div(CONV_OUTER_BLOCK_SIZE_K, input_shape.depth) * filter_chunks.depth * batch_size
	);
	//dim3 blocks_per_grid(1, 1, batch_size);

	for (int filter = 0; filter < input_shape.depth; filter++) {
		/*d_filter_outer_convolve_2d<<<blocks_per_grid, threads_per_block>>>(
			d_input,
			&d_filter[filter_shape.size() * filter], 
			d_output, 
			input_shape, 
			filter_shape, 
			output_shape, 
			filter
		);*/
		d_filter_outer_convolve_2d<<<blocks_per_grid, threads_per_block>>>(
			d_input,
			&d_filter[filter_shape.size() * filter],
			d_output,
			input_shape,
			filter_shape,
			filter_chunks,
			output_shape,
			padding,
			ceil_div(CONV_OUTER_BLOCK_SIZE_K, input_shape.depth),
			filter,
			batch_size
		);
	}
}


void filter_convolve_2d_derivative(
	float * d_input, 
	float * d_pds, 
	float * d_output, 
	shape input_shape, 
	shape pd_shape, 
	shape output_shape, 
	shape padding,
	size_t batch_size)
{
	shape filter_chunks = shape(
		ceil_div(FILTER_BLOCK_SIZE_N, pd_shape.width),
		ceil_div(FILTER_BLOCK_SIZE_M, pd_shape.height),
		ceil_div(FILTER_BLOCK_SIZE_K, pd_shape.depth)
	);	

	dim3 threads_per_block(CONV_DER_THREAD_SIZE_N, CONV_DER_THREAD_SIZE_M, CONV_DER_THREAD_SIZE_K);
	dim3 blocks_per_grid(
		ceil_div(CONV_DER_BLOCK_SIZE_N, input_shape.width) * filter_chunks.width,
		ceil_div(CONV_DER_BLOCK_SIZE_M, input_shape.height) * filter_chunks.height,
		ceil_div(CONV_DER_BLOCK_SIZE_K, input_shape.depth) * filter_chunks.depth * batch_size
	);

	d_filter_convolve_2d_derivative<<<blocks_per_grid, threads_per_block>>>(
		d_input,
		d_pds,
		d_output,
		input_shape,
		pd_shape,
		filter_chunks,
		output_shape,
		padding,
		ceil_div(CONV_DER_BLOCK_SIZE_K, input_shape.depth),
		batch_size
	);
	
	/*dim3 threads_per_block(CONV_DER_THREAD_SIZE_N, CONV_DER_THREAD_SIZE_M, CONV_DER_THREAD_SIZE_K);
	dim3 blocks_per_grid(output_shape.width, output_shape.height, batch_size);

	size_t max_dim = max(pd_shape.width, pd_shape.height);
	if (max_dim <= 1) {
		for (int filter = 0; filter < pd_shape.depth; filter++) {
			d_filter_convolve_2d_derivative<1><<<blocks_per_grid, threads_per_block>>> (
				d_input,
				d_pds,
				&d_output[filter * output_shape.size()],
				input_shape,
				pd_shape,
				output_shape,
				filter
			);
		}
	}
	else if (max_dim <= 2) {
		for (int filter = 0; filter < pd_shape.depth; filter++) {
			d_filter_convolve_2d_derivative<2><<<blocks_per_grid, threads_per_block>>> (
				d_input,
				d_pds,
				&d_output[filter * output_shape.size()],
				input_shape,
				pd_shape,
				output_shape,
				filter
			);
		}
	}
	else if (max_dim <= 4) {
		for (int filter = 0; filter < pd_shape.depth; filter++) {
			d_filter_convolve_2d_derivative<4><<<blocks_per_grid, threads_per_block>>> (
				d_input,
				d_pds,
				&d_output[filter * output_shape.size()],
				input_shape,
				pd_shape,
				output_shape,
				filter
			);
		}
	}
	else if (max_dim <= 8) {
		for (int filter = 0; filter < pd_shape.depth; filter++) {
			d_filter_convolve_2d_derivative<8><<<blocks_per_grid, threads_per_block>>> (
				d_input,
				d_pds,
				&d_output[filter * output_shape.size()],
				input_shape,
				pd_shape,
				output_shape,
				filter
			);
		}
	}
	else if (max_dim <= 16) {
		for (int filter = 0; filter < pd_shape.depth; filter++) {
			d_filter_convolve_2d_derivative<16><<<blocks_per_grid, threads_per_block>>> (
				d_input,
				d_pds,
				&d_output[filter * output_shape.size()],
				input_shape,
				pd_shape,
				output_shape,
				filter
			);
		}
	}
	else if (max_dim <= 32) {
		for (int filter = 0; filter < pd_shape.depth; filter++) {
			d_filter_convolve_2d_derivative<32><<<blocks_per_grid, threads_per_block>>> (
				d_input,
				d_pds,
				&d_output[filter * output_shape.size()],
				input_shape,
				pd_shape,
				output_shape,
				filter
			);
		}
	}
	else {
		throw new exception("Convolution image size currently not handled (too large)");
	}*/
}

void pool_2d_derivative(
	float * d_input, 
	int * d_mask, 
	float * d_output, 
	shape input_shape,
	shape output_shape, 
	size_t batch_size)
{
	size_t in_size = input_shape.width * input_shape.height;
	size_t out_size = output_shape.width * output_shape.height;
	dim3 threads_per_block(in_size, 1);
	dim3 blocks_per_grid(1, batch_size * output_shape.depth);
	if (in_size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, in_size);
	}
	d_pool_2d_derivative<<<blocks_per_grid, threads_per_block>>>(d_input, d_mask, d_output, in_size, out_size, batch_size * output_shape.depth);
}
