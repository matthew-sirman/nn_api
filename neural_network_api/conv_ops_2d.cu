#include "conv_ops_2d.h"

namespace nnet {
	namespace nnet_internal {
		//kernel function for forward pass convolutions
		__global__ void d_filter_convolve_2d(
			float* input,
			float* filter,
			float* output,
			shape input_shape,
			shape filter_shape,
			shape filter_chunks,
			shape output_shape,
			shape padding,
			int filter_no,
			int n_filters,
			size_t batch_size) {

			//declare shared memory for the filter and the loaded block. Copying to shared memory reduces overall data loads
			__shared__ float s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * FILTER_BLOCK_SIZE_K];
			__shared__ float s_load_block[(CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K * CONV_BATCH_DEPTH];

			//3d thread ids. Each element is cached in the registers to make accesses quicker
			int n_id = threadIdx.x;
			int m_id = threadIdx.y;
			int k_id = threadIdx.z;

			//3d filter tile ids. Each filter tile is loaded in a different thread block.
			int f_tile_n = blockIdx.x % filter_chunks.width * FILTER_BLOCK_SIZE_N;
			int f_tile_m = blockIdx.y % filter_chunks.height * FILTER_BLOCK_SIZE_M;
			int f_tile_k = blockIdx.z % filter_chunks.depth * FILTER_BLOCK_SIZE_K;

			//3d block tile ids. Each block tile is loaded in a different thread block
			int b_tile_n = (blockIdx.x / filter_chunks.width) * CONV_BLOCK_SIZE_N;
			int b_tile_m = (blockIdx.y / filter_chunks.height) * CONV_BLOCK_SIZE_M;
			int b_tile_k = f_tile_k;

			//the current element in the batch
			int b_elem = (blockIdx.z / filter_chunks.depth) * CONV_BATCH_DEPTH;

			//load filter chunk into shared memory

#pragma unroll
			for (int f_load_k = 0; f_load_k < FILTER_BLOCK_SIZE_K; f_load_k += CONV_THREAD_SIZE_K) {
				//indices for the k dimension for the shared and global strides
				int load_k_id = f_load_k + k_id;
				int g_load_k_id = f_tile_k + load_k_id;

#pragma unroll
				for (int f_load_m = 0; f_load_m < FILTER_BLOCK_SIZE_M; f_load_m += CONV_THREAD_SIZE_M) {
					//indices for the m dimension for the shared and global strides
					int load_m_id = f_load_m + m_id;
					int g_load_m_id = f_tile_m + load_m_id;

#pragma unroll
					for (int f_load_n = 0; f_load_n < FILTER_BLOCK_SIZE_N; f_load_n += CONV_THREAD_SIZE_N) {
						//indices for the n dimension for the shared and global strides
						int load_n_id = f_load_n + n_id;
						int g_load_n_id = f_tile_n + load_n_id;

						//the index in shared memory for the current load element
						int s_index = load_k_id * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M +
							load_m_id * FILTER_BLOCK_SIZE_N +
							load_n_id;

						//check that the global element is within the right range (to avoid illegal addressing)
						//check that the batch element is within range
						if (g_load_n_id < filter_shape.width &&
							g_load_m_id < filter_shape.height &&
							g_load_k_id < filter_shape.depth &&
							b_elem < batch_size) {

							//the index in global memory for the current load element
							int g_index = g_load_k_id * filter_shape.width * filter_shape.height +
								g_load_m_id * filter_shape.width +
								g_load_n_id;

							//load the element into shared memory
							s_filter[s_index] = filter[g_index];
						}
						else {
							//if the global element is outside the range set the shared element to 0
							s_filter[s_index] = 0.0;
						}
					}
				}
			}

			//synchronise the threads in this block (to complete filter loading)
			cuda_syncthreads();

			//load input chunk

#pragma unroll
			for (int batch = 0; batch < CONV_BATCH_DEPTH; batch++) {
#pragma unroll
				for (int b_load_k = 0; b_load_k < CONV_BLOCK_SIZE_K; b_load_k += CONV_THREAD_SIZE_K) {
					//indices for the k dimension for the shared and global strides
					int load_k_id = b_load_k + k_id;
					int g_load_k_id = b_tile_k + load_k_id;

#pragma unroll
					for (int b_load_m = 0; b_load_m < CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M; b_load_m += CONV_THREAD_SIZE_M) {
						//indices for the m dimension for the shared and global strides. The global element is padded
						int load_m_id = b_load_m + m_id;
						int g_load_m_id = f_tile_m + b_tile_m + load_m_id - padding.height;

#pragma unroll
						for (int b_load_n = 0; b_load_n < CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N; b_load_n += CONV_THREAD_SIZE_N) {
							//indices for the n dimension for the shared and global strides. The global element is padded
							int load_n_id = b_load_n + n_id;
							int g_load_n_id = f_tile_n + b_tile_n + load_n_id - padding.width;

							//the index in shared memory for the current load element
							int s_index = batch * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K +
								load_k_id * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) +
								load_m_id * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) +
								load_n_id;

							//check that the global element is within the right range (to avoid illegal addressing)
							//check that the batch element is within range
							if (0 <= g_load_n_id &&
								g_load_n_id < input_shape.width &&
								0 <= g_load_m_id &&
								g_load_m_id < input_shape.height &&
								g_load_k_id < input_shape.depth &&
								b_elem + batch < batch_size) {

								//the index in global memory for the current load element
								int g_index = (b_elem + batch) * input_shape.width * input_shape.height * input_shape.depth +
									g_load_k_id * input_shape.width * input_shape.height +
									g_load_m_id * input_shape.width +
									g_load_n_id;

								//load the element into shared memory
								s_load_block[s_index] = input[g_index];
							}
							else {
								//if the global element is outside the range set the shared element to 0
								s_load_block[s_index] = 0.0;
							}
						}
					}
				}
			}

			//synchronise the threads in this block (to complete block loading)
			cuda_syncthreads();

			//convolve and write

#pragma unroll
			for (int batch = 0; batch < CONV_BATCH_DEPTH; batch++) {
				if (k_id == 0) {
#pragma unroll
					for (int stride_m = 0; stride_m < CONV_THREAD_BLOCK_M; stride_m++) {
						//stride index for the m direction
						int start_m = m_id * CONV_THREAD_BLOCK_M + stride_m;
#pragma unroll
						for (int stride_n = 0; stride_n < CONV_THREAD_BLOCK_N; stride_n++) {
							//stride index for the n direction
							int start_n = n_id * CONV_THREAD_BLOCK_N + stride_n;

							//get the output indices for writing this element
							int out_n_id = b_tile_n + start_n;
							int out_m_id = b_tile_m + start_m;
							int out_k_id = (b_elem + batch) * n_filters + filter_no;

							//check that this element is within the range of the output shape
							//to avoid illegal addressing
							if (out_n_id < output_shape.width &&
								out_m_id < output_shape.height &&
								b_elem + batch < batch_size) {

								//the writing address
								int out_index = out_k_id * output_shape.width * output_shape.height +
									out_m_id * output_shape.width +
									out_n_id;

								//calculate one dot product and caches it
								float inc = calculate_conv2d_dot<CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N, CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M, CONV_THREAD_BLOCK_K>(
									s_filter,
									&s_load_block[batch * (CONV_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K],
									start_n,
									start_m
									);

								//accumulate the dot product to the output memory at the writing address
								//this is an atomic operation to avoid memory races
								atomic_add(&output[out_index], inc);
							}
						}
					}
				}
			}
		}

		//kernel function for backward pass convolutions
		__global__ void d_filter_outer_convolve_2d(
			float* input,
			float* filter,
			float* output,
			shape input_shape,
			shape filter_shape,
			shape filter_chunks,
			shape output_shape,
			shape padding,
			int filter_no,
			size_t batch_size) {

			//declare shared memory for the filter and the loaded block. Copying to shared memory reduces overall data loads
			__shared__ float s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * FILTER_BLOCK_SIZE_K];
			__shared__ float s_load_block[(CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_OUTER_BLOCK_SIZE_K];

			//3d thread ids. Each element is cached in the registers to make accesses quicker
			int n_id = threadIdx.x;
			int m_id = threadIdx.y;
			int k_id = threadIdx.z;

			//3d filter tile ids. Each filter tile is loaded in a different thread block.
			int f_tile_n = blockIdx.x % filter_chunks.width * FILTER_BLOCK_SIZE_N;
			int f_tile_m = blockIdx.y % filter_chunks.height * FILTER_BLOCK_SIZE_M;
			int f_tile_k = blockIdx.z % filter_chunks.depth * FILTER_BLOCK_SIZE_K;

			//3d block tile ids. Each block tile is loaded in a different thread block
			int b_tile_n = blockIdx.x / filter_chunks.width * CONV_OUTER_BLOCK_SIZE_N;
			int b_tile_m = blockIdx.y / filter_chunks.height * CONV_OUTER_BLOCK_SIZE_M;

			//the current element in the batch
			int b_elem = (blockIdx.z / filter_chunks.depth) * CONV_OUTER_BLOCK_SIZE_K;

			//load filter chunk

#pragma unroll
			for (int f_load_k = 0; f_load_k < FILTER_BLOCK_SIZE_K; f_load_k += CONV_OUTER_THREAD_SIZE_K) {
				//indices for the k dimension for the shared and global strides
				int load_k_id = f_load_k + k_id;
				int g_load_k_id = f_tile_k + load_k_id;

#pragma unroll
				for (int f_load_m = 0; f_load_m < FILTER_BLOCK_SIZE_M; f_load_m += CONV_OUTER_THREAD_SIZE_M) {
					//indices for the m dimension for the shared and global strides
					int load_m_id = f_load_m + m_id;
					int g_load_m_id = f_tile_m + load_m_id;

#pragma unroll
					for (int f_load_n = 0; f_load_n < FILTER_BLOCK_SIZE_N; f_load_n += CONV_OUTER_THREAD_SIZE_N) {
						//indices for the n dimension for the shared and global strides
						int load_n_id = f_load_n + n_id;
						int g_load_n_id = f_tile_n + load_n_id;

						//the index in shared memory for the current load element
						int s_index = load_k_id * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M +
							load_m_id * FILTER_BLOCK_SIZE_N +
							load_n_id;

						//check that the global element is within the right range (to avoid illegal addressing)
						//check that the batch element is within range
						if (g_load_n_id < filter_shape.width &&
							g_load_m_id < filter_shape.height &&
							g_load_k_id < filter_shape.depth) {

							//the index in global memory for the current load element
							int g_index = g_load_k_id * filter_shape.width * filter_shape.height +
								(filter_shape.height - 1 - g_load_m_id) * filter_shape.width +
								(filter_shape.width - 1 - g_load_n_id);

							//load the element into shared memory
							s_filter[s_index] = filter[g_index];
						}
						else {
							//if the global element is outside the range set the shared element to 0
							s_filter[s_index] = 0.0;
						}
					}
				}
			}

			//synchronise the threads in this block (to complete filter loading)
			cuda_syncthreads();

			//load input block + edge

#pragma unroll
			for (int b_load_k = 0; b_load_k < CONV_OUTER_BLOCK_SIZE_K; b_load_k += CONV_OUTER_THREAD_SIZE_K) {
				//indices for the k dimension for the shared and global strides
				int load_k_id = b_load_k + k_id;
				int g_load_k_id = b_elem + load_k_id;

#pragma unroll
				for (int b_load_m = 0; b_load_m < CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M; b_load_m += CONV_OUTER_THREAD_SIZE_M) {
					//indices for the m dimension for the shared and global strides
					int load_m_id = b_load_m + m_id;
					int g_load_m_id = b_tile_m + load_m_id - filter_shape.height + f_tile_m + 1 + padding.height;

#pragma unroll
					for (int b_load_n = 0; b_load_n < CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N; b_load_n += CONV_OUTER_THREAD_SIZE_N) {
						//indices for the n dimension for the shared and global strides
						int load_n_id = b_load_n + n_id;
						int g_load_n_id = b_tile_n + load_n_id - filter_shape.width + f_tile_n + 1 + padding.width;

						//the index in shared memory for the current load element
						int s_index = load_k_id * (CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) +
							load_m_id * (CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) +
							load_n_id;

						//check that the global element is within the right range (to avoid illegal addressing)
						//check that the batch element is within range
						if (0 <= g_load_n_id &&
							g_load_n_id < input_shape.width &&
							0 <= g_load_m_id &&
							g_load_m_id < input_shape.height &&
							g_load_k_id < batch_size) {

							//the index in global memory for the current load element
							int g_index = g_load_k_id * input_shape.width * input_shape.height * input_shape.depth +
								filter_no * input_shape.width * input_shape.height +
								g_load_m_id * input_shape.width +
								g_load_n_id;

							//load the element into shared memory
							s_load_block[s_index] = input[g_index];
						}
						else {
							//if the global element is outside the range set the shared element to 0
							s_load_block[s_index] = 0.0;
						}
					}
				}
			}

			//synchronise the threads in this block (to complete block loading)
			cuda_syncthreads();

			//convole and accumulate to output

#pragma unroll
			for (int f_k = 0; f_k < FILTER_BLOCK_SIZE_K; f_k++) {
				//stride index for the filter
				int out_k_id = f_tile_k + f_k;

#pragma unroll
				for (int b_k = 0; b_k < CONV_OUTER_THREAD_BLOCK_K; b_k++) {
					//stride index for the k direction
					int b_k_id = k_id * CONV_OUTER_THREAD_BLOCK_K + b_k;

#pragma unroll
					for (int b_m = 0; b_m < CONV_OUTER_THREAD_BLOCK_M; b_m++) {
						//stride index for the m direction
						int b_m_id = m_id * CONV_OUTER_THREAD_BLOCK_M + b_m;
						int out_m_id = b_tile_m + b_m_id;

#pragma unroll
						for (int b_n = 0; b_n < CONV_OUTER_THREAD_BLOCK_N; b_n++) {
							//stride index for the n direction
							int b_n_id = n_id * CONV_OUTER_THREAD_BLOCK_N + b_n;
							int out_n_id = b_tile_n + b_n_id;

							//check that this element is within the range of the output shape
							//to avoid illegal addressing
							if (out_n_id < output_shape.width &&
								out_m_id < output_shape.height &&
								out_k_id < output_shape.depth &&
								b_elem < batch_size) {

								//the writing address
								int out_index = (b_elem + b_k_id) * output_shape.width * output_shape.height * output_shape.depth +
									out_k_id * output_shape.width * output_shape.height +
									out_m_id * output_shape.width +
									out_n_id;

								//calculate one dot product and caches it
								float inc = calculate_conv2d_dot<CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N, CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M, 1>(
									&s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * f_k],
									&s_load_block[(CONV_OUTER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_OUTER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * b_k],
									b_n_id,
									b_m_id
									);

								//accumulate the dot product to the output memory at the writing address
								//this is an atomic operation to avoid memory races
								atomic_add(&output[out_index], inc);
							}
						}
					}
				}
			}
		}

		//kernel function for training convolutional filters
		__global__ void d_filter_convolve_2d_derivative(
			float* input,
			float* filter,
			float* output,
			shape input_shape,
			shape filter_shape,
			shape filter_chunks,
			shape output_shape,
			shape padding,
			int input_depth,
			size_t batch_size) {

			//declare shared memory for the filter and the loaded block. Copying to shared memory reduces overall data loads
			__shared__ float s_filter[FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M * FILTER_BLOCK_SIZE_K];
			__shared__ float s_load_block[(CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) * CONV_BLOCK_SIZE_K];

			//3d thread ids. Each element is cached in the registers to make accesses quicker
			int n_id = threadIdx.x;
			int m_id = threadIdx.y;
			int k_id = threadIdx.z;

			//3d filter tile ids. Each filter tile is loaded in a different thread block.
			int f_tile_n = blockIdx.x % filter_chunks.width * FILTER_BLOCK_SIZE_N;
			int f_tile_m = blockIdx.y % filter_chunks.height * FILTER_BLOCK_SIZE_M;
			int f_tile_k = blockIdx.z % filter_chunks.depth * FILTER_BLOCK_SIZE_K;

			//3d block tile ids. Each block tile is loaded in a different thread block
			int b_tile_n = (blockIdx.x / filter_chunks.width) * CONV_DER_BLOCK_SIZE_N;
			int b_tile_m = (blockIdx.y / filter_chunks.height) * CONV_DER_BLOCK_SIZE_M;
			int b_tile_k = (blockIdx.z / filter_chunks.depth) % input_depth * CONV_DER_BLOCK_SIZE_K;

			//the current element in the batch
			int b_elem = blockIdx.z / (filter_chunks.depth * input_depth);

			//load filter chunk

#pragma unroll
			for (int f_load_k = 0; f_load_k < FILTER_BLOCK_SIZE_K; f_load_k += CONV_DER_THREAD_SIZE_K) {
				//indices for the k dimension for the shared and global strides
				int load_k_id = f_load_k + k_id;
				int g_load_k_id = f_tile_k + load_k_id;

#pragma unroll
				for (int f_load_m = 0; f_load_m < FILTER_BLOCK_SIZE_M; f_load_m += CONV_DER_THREAD_SIZE_M) {
					//indices for the m dimension for the shared and global strides
					int load_m_id = f_load_m + m_id;
					int g_load_m_id = f_tile_m + load_m_id;

#pragma unroll
					for (int f_load_n = 0; f_load_n < FILTER_BLOCK_SIZE_N; f_load_n += CONV_DER_THREAD_SIZE_N) {
						//indices for the n dimension for the shared and global strides
						int load_n_id = f_load_n + n_id;
						int g_load_n_id = f_tile_n + load_n_id;

						//the index in shared memory for the current load element
						int s_index = load_k_id * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M +
							load_m_id * FILTER_BLOCK_SIZE_N +
							load_n_id;

						//check that the global element is within the right range (to avoid illegal addressing)
						//check that the batch element is within range
						if (g_load_n_id < filter_shape.width &&
							g_load_m_id < filter_shape.height &&
							g_load_k_id < filter_shape.depth) {

							//the index in global memory for the current load element
							int g_index = b_elem * filter_shape.width * filter_shape.height * filter_shape.depth +
								g_load_k_id * filter_shape.width * filter_shape.height +
								g_load_m_id * filter_shape.width +
								g_load_n_id;

							//load the element into shared memory
							s_filter[s_index] = filter[g_index];
						}
						else {
							//if the global element is outside the range set the shared element to 0
							s_filter[s_index] = 0.0;
						}
					}
				}
			}

			//synchronise the threads in this block (to complete filter loading)
			cuda_syncthreads();

			//load input chunk

#pragma unroll
			for (int b_load_k = 0; b_load_k < CONV_DER_BLOCK_SIZE_K; b_load_k += CONV_DER_THREAD_SIZE_K) {
				//indices for the k dimension for the shared and global strides
				int load_k_id = b_load_k + k_id;
				int g_load_k_id = b_tile_k + load_k_id;

#pragma unroll
				for (int b_load_m = 0; b_load_m < CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M; b_load_m += CONV_DER_THREAD_SIZE_M) {
					//indices for the m dimension for the shared and global strides
					int load_m_id = b_load_m + m_id;
					int g_load_m_id = f_tile_m + b_tile_m + load_m_id - padding.height;

#pragma unroll
					for (int b_load_n = 0; b_load_n < CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N; b_load_n += CONV_DER_THREAD_SIZE_N) {
						//indices for the n dimension for the shared and global strides
						int load_n_id = b_load_n + n_id;
						int g_load_n_id = f_tile_n + b_tile_n + load_n_id - padding.width;

						//the index in shared memory for the current load element
						int s_index = load_k_id * (CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M) +
							load_m_id * (CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) +
							load_n_id;

						//check that the global element is within the right range (to avoid illegal addressing)
						//check that the batch element is within range
						if (0 <= g_load_n_id &&
							g_load_n_id < input_shape.width &&
							0 <= g_load_m_id &&
							g_load_m_id < input_shape.height &&
							g_load_k_id < input_shape.depth) {

							//the index in global memory for the current load element
							int g_index = b_elem * input_shape.width * input_shape.height * input_shape.depth +
								g_load_k_id * input_shape.width * input_shape.height +
								g_load_m_id * input_shape.width +
								g_load_n_id;

							//load the element into shared memory
							s_load_block[s_index] = input[g_index];
						}
						else {
							//if the global element is outside the range set the shared element to 0
							s_load_block[s_index] = 0.0;
						}
					}
				}
			}

			//synchronise the threads in this block (to complete block loading)
			cuda_syncthreads();

			//convolve and write

#pragma unroll
			for (int filter_k = 0; filter_k < FILTER_BLOCK_SIZE_K; filter_k++) {
#pragma unroll
				for (int stride_k = 0; stride_k < CONV_DER_THREAD_BLOCK_K; stride_k++) {
					//stride index for the k direction			
					int start_k = k_id * CONV_DER_THREAD_BLOCK_K + stride_k;
#pragma unroll
					for (int stride_m = 0; stride_m < CONV_DER_THREAD_BLOCK_M; stride_m++) {
						//stride index for the m direction
						int start_m = m_id * CONV_DER_THREAD_BLOCK_M + stride_m;
#pragma unroll
						for (int stride_n = 0; stride_n < CONV_DER_THREAD_BLOCK_N; stride_n++) {
							//stride index for the n direction
							int start_n = n_id * CONV_DER_THREAD_BLOCK_N + stride_n;

							//writing indices
							int out_n_id = b_tile_n + start_n;
							int out_m_id = b_tile_m + start_m;
							int out_layer_id = b_tile_k + start_k;
							int out_filter_id = f_tile_k + filter_k;
							int out_k_id = out_filter_id * output_shape.depth + out_layer_id;

							//check that the output element is within range to avoid illegal
							//addressing
							if (out_n_id < output_shape.width &&
								out_m_id < output_shape.height &&
								out_filter_id < filter_shape.depth &&
								out_layer_id < output_shape.depth) {

								//the output index
								int out_index = out_k_id * output_shape.width * output_shape.height +
									out_m_id * output_shape.width +
									out_n_id;

								//calculate one dot product and caches it
								float inc = calculate_conv2d_dot<CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N, CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M, 1>(
									&s_filter[filter_k * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M],
									&s_load_block[start_k * (CONV_DER_BLOCK_SIZE_N + FILTER_BLOCK_SIZE_N) * (CONV_DER_BLOCK_SIZE_M + FILTER_BLOCK_SIZE_M)],
									start_n,
									start_m
									);

								//accumulate the dot product to the output memory at the writing address
								//this is an atomic operation to avoid memory races
								atomic_add(&output[out_index], inc);
							}
						}
					}
				}
			}

			cuda_syncthreads();
		}

		//kernel function for forward pass pooling
		__global__ void d_pool_2d(
			float* input,
			int* mask,
			float* output,
			shape input_shape,
			shape pool_size,
			shape stride,
			shape output_shape,
			shape padding,
			size_t batch_size) {

			//declare shared block to convolve over taking the max at each point
			__shared__ float s_block[(POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) * (POOL_BLOCK_SIZE_M + MAX_POOL_SIZE) * POOL_BLOCK_DEPTH];

			//3d thread ids. Each element is cached in the registers to make accesses quicker
			int n_id = threadIdx.x;
			int m_id = threadIdx.y;
			int k_id = threadIdx.z;

			//3d filter tile ids. Each filter tile is loaded in a different thread block.
			int n_tile = blockIdx.x * POOL_BLOCK_SIZE_N;
			int m_tile = blockIdx.y * POOL_BLOCK_SIZE_M;
			int k_tile = blockIdx.z * POOL_BLOCK_DEPTH;

			//load image block into s_block
			//each block convolves one region of the image

#pragma unroll
			for (int load_k = 0; load_k < POOL_BLOCK_DEPTH; load_k += POOL_THREAD_SIZE_K) {
				//indices for the k dimension for the shared and global strides
				int load_k_id = load_k + k_id;

				int g_load_k_id = (load_k_id + k_tile) % input_shape.depth;
				int g_load_elem_id = ((load_k_id + k_tile) / input_shape.depth);
				int g_in_elem_id = g_load_elem_id * (input_shape.width * input_shape.height * input_shape.depth);

				int g_out_elem_id = g_load_elem_id * (output_shape.width * output_shape.height * output_shape.depth);

#pragma unroll
				for (int load_m = 0; load_m < POOL_BLOCK_SIZE_M + MAX_POOL_SIZE; load_m += POOL_THREAD_SIZE_M) {
					//indices for the m dimension for the shared and global strides
					int load_m_id = load_m + m_id;
					int g_load_m_id = load_m_id + m_tile;

#pragma unroll
					for (int load_n = 0; load_n < POOL_BLOCK_SIZE_N + MAX_POOL_SIZE; load_n += POOL_THREAD_SIZE_N) {
						//indices for the n dimension for the shared and global strides
						int load_n_id = load_n + n_id;
						int g_load_n_id = load_n_id + n_tile;

						//the shared memory index for this specific element
						int s_load_id = load_k_id * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) * (POOL_BLOCK_SIZE_M + MAX_POOL_SIZE) + load_m_id * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) + load_n_id;

						//check that the element is within range to avoid illegal addressing
						if (g_load_n_id < input_shape.width &&
							g_load_m_id < input_shape.height &&
							g_load_elem_id < batch_size) {

							//the global memory index for this specific element
							int g_load_id = g_in_elem_id + g_load_k_id * input_shape.width * input_shape.height + g_load_m_id * input_shape.width + g_load_n_id;

							//load the element from global memory into shared memory
							s_block[s_load_id] = input[g_load_id];
						}
						else {
							//if the element is outside of range load the minimum negative number instead 
							//so that it doesn't get chosen by the max pool
							s_block[s_load_id] = FLOAT_MIN;
						}
					}
				}

				//synchronise the threads to complete block loading
				cuda_syncthreads();

				//loop through each stride with even divisions among threads
				//each thread iterates through every pool with the top left corner in the thread's allocated block

				//calculate offsets so that the different blocks start at the right point (as it will not always be perfectly left aligned)
				int n_offset = ((input_shape.width + padding.width - pool_size.width) - (n_tile + n_id * POOL_THREAD_BLOCK_N)) % stride.width;
				int m_offset = ((input_shape.height + padding.height - pool_size.height) - (m_tile + m_id * POOL_THREAD_BLOCK_M)) % stride.height;

				//stride over the image and calculate the max values for each pool region
				for (int stride_m = m_offset; stride_m < POOL_THREAD_BLOCK_M; stride_m += stride.height) {
					for (int stride_n = n_offset; stride_n < POOL_THREAD_BLOCK_N; stride_n += stride.width) {
						//indices of the current stride for this thread
						int stride_index_n = n_id * POOL_THREAD_BLOCK_N + stride_n;
						int stride_index_m = m_id * POOL_THREAD_BLOCK_M + stride_m;
						//initialise the max value to the minimum negative number (such that each pool value should be larger)
						float tmp_max = FLOAT_MIN;
						//cache indices of the greatest element for masking
						int n_index = stride_index_n;
						int m_index = stride_index_m;
						for (int pool_m = 0; pool_m < pool_size.height; pool_m++) {
							for (int pool_n = 0; pool_n < pool_size.width; pool_n++) {
								//cache the index to compare without double loading
								float tmp_read = s_block[load_k_id * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) * (POOL_BLOCK_SIZE_M + MAX_POOL_SIZE) +
									(stride_index_m + pool_m) * (POOL_BLOCK_SIZE_N + MAX_POOL_SIZE) +
									stride_index_n + pool_n];
								//if this element is greater than the current largest
								if (tmp_read > tmp_max) {
									//update the current largest
									tmp_max = tmp_read;
									//set the indices to tis element
									n_index = stride_index_n + pool_n;
									m_index = stride_index_m + pool_m;
								}
							}
						}

						//write tmp_max to output

						//out indices for writing
						int g_out_n = (n_tile + stride_index_n) / stride.width;
						int g_out_m = (m_tile + stride_index_m) / stride.height;

						//the index of the largest element for the mask
						int mask_out_index = (m_tile + m_index) * input_shape.width + (n_tile + n_index);

						//check that the element is within the range of the output to avoid illegal addressing
						if (g_out_n < output_shape.width && g_out_m < output_shape.height && g_load_elem_id < batch_size) {
							//write output element to output array
							output[g_out_elem_id + g_load_k_id * output_shape.width * output_shape.height + g_out_m * output_shape.width + g_out_n] = tmp_max;
							//write mask element to mask array
							mask[g_out_elem_id + g_load_k_id * output_shape.width * output_shape.height + g_out_m * output_shape.width + g_out_n] = mask_out_index;
						}
					}
				}
			}
		}

		//kernel function for backward pass pooling
		__global__ void d_pool_2d_derivative(
			float* input,
			int* mask,
			float* output,
			size_t input_size,
			size_t output_size,
			size_t batch_size) {

			//the thread index for this specific thread
			int t_id = threadIdx.x + blockIdx.x * blockDim.x;

			//the batch index for this block
			int batch_index = blockIdx.y;

			//check that the index is within range to avoid illegal addressing
			if (t_id < input_size && batch_index < batch_size) {
				//get the index to write out to which we stored in the mask during forward propagation
				int out_id = mask[batch_index * input_size + t_id];
				//accumulate the partial derivative value to the output index from the mask atomically
				atomic_add(&output[batch_index * output_size + out_id], input[batch_index * input_size + t_id]);
			}
		}

		template <int BLOCK_N, int BLOCK_M, int DEPTH>
		__device__ float calculate_conv2d_dot(volatile float* s_filter, volatile float* s_load_block, int start_n, int start_m)
		{
			//register accumulator variable
			float accum = 0;

			//stride through the region of interest
#pragma unroll
			for (int dot_k = 0; dot_k < DEPTH; dot_k++) {
#pragma unroll
				for (int dot_m = 0; dot_m < FILTER_BLOCK_SIZE_M; dot_m++) {
#pragma unroll
					for (int dot_n = 0; dot_n < FILTER_BLOCK_SIZE_N; dot_n++) {
						//get the indices of the filter and the block
						int f_index = dot_k * FILTER_BLOCK_SIZE_N * FILTER_BLOCK_SIZE_M + dot_m * FILTER_BLOCK_SIZE_N + dot_n;
						int b_index = dot_k * BLOCK_N * BLOCK_M + (dot_m + start_m) * BLOCK_N + (dot_n + start_n);

						//increment the accumulator by the product of the filter and block indices
						accum += s_filter[f_index] * s_load_block[b_index];
					}
				}
			}

			//return the accumulated sum (the dot product)
			return accum;
		}

		void filter_convolve_2d(
			float* d_input,
			float* d_filter,
			float* d_output,
			shape input_shape,
			shape output_shape,
			shape filter_shape,
			shape padding,
			size_t batch_size)
		{
			//divide the filter into chunks
			shape filter_chunks = shape(
				ceil_div(FILTER_BLOCK_SIZE_N, filter_shape.width),
				ceil_div(FILTER_BLOCK_SIZE_M, filter_shape.height),
				ceil_div(FILTER_BLOCK_SIZE_K, filter_shape.depth)
			);

			//declare the thread size of each block
			dim3 threads_per_block(CONV_THREAD_SIZE_N, CONV_THREAD_SIZE_M, CONV_THREAD_SIZE_K);

			//declare the number of blocks in each dimension
			dim3 blocks_per_grid(
				ceil_div(CONV_BLOCK_SIZE_N, input_shape.width + padding.width) * filter_chunks.width,
				ceil_div(CONV_BLOCK_SIZE_M, input_shape.height + padding.width) * filter_chunks.height,
				ceil_div(CONV_BATCH_DEPTH, batch_size) * filter_chunks.depth
			);

			//loop through each filter and launch the convolution kernel for each
			for (int filter = 0; filter < output_shape.depth; filter++)
			{
				d_filter_convolve_2d << <blocks_per_grid, threads_per_block >> > (
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

		void max_pool_2d(
			float* d_input,
			int* d_mask,
			float* d_output,
			shape input_shape,
			shape pool_size,
			shape stride,
			shape output_shape,
			shape padding,
			size_t batch_size)
		{
			//declare the thread size of each block
			dim3 threads_per_block(POOL_THREAD_SIZE_N, POOL_THREAD_SIZE_M, POOL_THREAD_SIZE_K);

			//declare the number of blocks in each dimension
			dim3 blocks_per_grid(
				ceil_div(POOL_BLOCK_SIZE_N, input_shape.width + padding.width),
				ceil_div(POOL_BLOCK_SIZE_M, input_shape.height + padding.height),
				ceil_div(POOL_BLOCK_DEPTH, input_shape.depth * batch_size)
			);

			//launch the pooling kernel
			d_pool_2d << <blocks_per_grid, threads_per_block >> > (d_input, d_mask, d_output, input_shape, pool_size, stride, output_shape, padding, batch_size);
		}

		void filter_outer_convolve_2d(
			float* d_input,
			float* d_filter,
			float* d_output,
			shape input_shape,
			shape output_shape,
			shape filter_shape,
			shape padding,
			size_t batch_size)
		{
			//divide the filter into chunks
			shape filter_chunks = shape(
				ceil_div(FILTER_BLOCK_SIZE_N, filter_shape.width),
				ceil_div(FILTER_BLOCK_SIZE_M, filter_shape.height),
				ceil_div(FILTER_BLOCK_SIZE_K, filter_shape.depth)
			);

			//declare the thread size of each block
			dim3 threads_per_block(CONV_OUTER_THREAD_SIZE_N, CONV_OUTER_THREAD_SIZE_M, CONV_OUTER_THREAD_SIZE_K);

			//declare the number of blocks in each dimension
			dim3 blocks_per_grid(
				ceil_div(CONV_OUTER_BLOCK_SIZE_N, input_shape.width) * filter_chunks.width,
				ceil_div(CONV_OUTER_BLOCK_SIZE_M, input_shape.height) * filter_chunks.height,
				ceil_div(CONV_OUTER_BLOCK_SIZE_K, batch_size) * filter_chunks.depth
			);

			//loop through each filter and launch the outer convolution kernel for each
			for (int filter = 0; filter < input_shape.depth; filter++) {
				d_filter_outer_convolve_2d << <blocks_per_grid, threads_per_block >> > (
					d_input,
					&d_filter[filter_shape.size() * filter],
					d_output,
					input_shape,
					filter_shape,
					filter_chunks,
					output_shape,
					padding,
					filter,
					batch_size
					);
			}
		}


		void filter_convolve_2d_derivative(
			float* d_input,
			float* d_pds,
			float* d_output,
			shape input_shape,
			shape pd_shape,
			shape output_shape,
			shape padding,
			size_t batch_size)
		{
			//divide the filter into chunks
			shape filter_chunks = shape(
				ceil_div(FILTER_BLOCK_SIZE_N, pd_shape.width),
				ceil_div(FILTER_BLOCK_SIZE_M, pd_shape.height),
				ceil_div(FILTER_BLOCK_SIZE_K, pd_shape.depth)
			);

			//declare the thread size of each block
			dim3 threads_per_block(CONV_DER_THREAD_SIZE_N, CONV_DER_THREAD_SIZE_M, CONV_DER_THREAD_SIZE_K);

			//declare the number of blocks in each dimension
			dim3 blocks_per_grid(
				ceil_div(CONV_DER_BLOCK_SIZE_N, input_shape.width) * filter_chunks.width,
				ceil_div(CONV_DER_BLOCK_SIZE_M, input_shape.height) * filter_chunks.height,
				ceil_div(CONV_DER_BLOCK_SIZE_K, input_shape.depth) * filter_chunks.depth * batch_size
			);

			//launch the convolutional derivative kernel
			d_filter_convolve_2d_derivative << <blocks_per_grid, threads_per_block >> > (
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
		}

		void max_pool_2d_derivative(
			float* d_input,
			int* d_mask,
			float* d_output,
			shape input_shape,
			shape output_shape,
			size_t batch_size)
		{
			//work out the total size of each input layer
			size_t in_size = input_shape.width * input_shape.height;
			//work out the total size of each output layer
			size_t out_size = output_shape.width * output_shape.height;

			//declare the number of threads per block
			dim3 threads_per_block(in_size, 1);

			//declare the number of blocks
			dim3 blocks_per_grid(1, batch_size * output_shape.depth);

			//if there are too many threads, split the threads among different blocks
			if (in_size > BLOCK_SIZE) {
				threads_per_block.x = BLOCK_SIZE;
				blocks_per_grid.x = ceil_div(BLOCK_SIZE, in_size);
			}

			//launch the pool derivative kernel
			d_pool_2d_derivative << <blocks_per_grid, threads_per_block >> > (d_input, d_mask, d_output, in_size, out_size, batch_size * output_shape.depth);
		}
		
	}
}