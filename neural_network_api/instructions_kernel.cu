#include "instructions_kernel.h"

__global__ void d_fast_copy(float *input, float *output, int size) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size)
		output[id] = input[id];
}

__global__ void d_staggered_copy(float *input, float *output, int in_size, int out_size) {
	unsigned int tid = threadIdx.x + BLOCK_SIZE * blockIdx.x;
	unsigned int inid = blockIdx.y;

	if (tid < in_size) {
		output[out_size * inid + tid] = input[in_size * inid + tid];
	}
	else {
		output[out_size * inid + tid] = 0;
	}
}

__global__ void d_random_scale_offset(float *arr, float scale, float offset, int size) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size)
		arr[id] = arr[id] * scale + offset;
}

__global__ void d_fill_array(float * arr, float value, int size) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size)
		arr[id] = value;
}

__global__ void d_add_matrices(float *input, float *biases, float *output, int size) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int nid = blockIdx.y;
	int off = nid * size;
	if (id < size)
		output[id + off] = input[id + off] + biases[id];
}

template <unsigned int block_size>
__global__ void d_multiply_weights(float *input, float *weights, float *output, int cols, int rows) {
	__shared__ float s_matmul[BLOCK_SIZE];
	//__shared__ float s_input0[BLOCK_SIZE], s_input1[BLOCK_SIZE];

	int tid = threadIdx.x;
	int i = 2 * block_size * blockIdx.x + tid;

	int rowId = blockIdx.y;
	int outId = blockIdx.z;

	int inOff = outId * cols;

	/*if (i + block_size < cols) {
		s_input0[tid] = input[i + inOff];
		s_input1[tid] = input[i + block_size + inOff];
	}
	else if (i < cols) {
		s_input0[tid] = input[i + inOff];
		s_input1[tid] = 0;
	}
	else {
		s_input0[tid] = 0;
		s_input1[tid] = 0;
	}

	cuda_syncthreads();

	s_matmul[tid] = weights[rowId * cols + i] * s_input0[tid] + weights[rowId * cols + i + block_size] * s_input1[tid];*/

	/*int gridSize = 2 * block_size * gridDim.x;
	s_matmul[tid] = 0;

	while (i < rows) {
		s_matmul[tid] += weights[rowId * cols + i] * input[i + inOff] + weights[rowId * cols + i + block_size] * input[i + block_size + inOff];
		i += gridSize;
	}*/

	if (i + block_size < cols) {
		s_matmul[tid] = weights[rowId * cols + i] * input[i + inOff] + weights[rowId * cols + i + block_size] * input[i + block_size + inOff];
	}
	else if (i < cols) {
		s_matmul[tid] = weights[rowId * cols + i] * input[i + inOff];
	}
	else {
		s_matmul[tid] = 0;
	}

	/*if (i + block_size < cols) {
		s_matmul[tid] = weights[rowId * cols + i] * input[i + inOff] + weights[rowId * cols + i + block_size] * input[i + block_size + inOff];
	}
	else if (i < cols) {
		s_matmul[tid] = weights[rowId * cols + i] * input[i + inOff];
	}
	else {
		s_matmul[tid] = 0;
	}*/

	cuda_syncthreads();
	
	if (block_size >= 512) {
		if (tid < 256) {
			s_matmul[tid] += s_matmul[tid + 256];
			cuda_syncthreads();
		}
	}
	if (block_size >= 256) {
		if (tid < 128) {
			s_matmul[tid] += s_matmul[tid + 128];
			cuda_syncthreads();
		}
	}
	if (block_size >= 128) {
		if (tid < 64) {
			s_matmul[tid] += s_matmul[tid + 64];
			cuda_syncthreads();
		}
	}

	if (tid < 32) {
		warp_reduce<block_size>(s_matmul, tid);
	}

	if (tid == 0) {
		if (blockIdx.x == 0)
			output[rowId + outId * rows] = s_matmul[0];
		else
			output[rowId + outId * rows] += s_matmul[0];
		//output[rowId + outId * rows] = s_matmul[0];
	}

	/*if ((start + tid) < cols)
		s_matmul[tid] = weights[rowId * cols + start + tid] * input[start + tid];
	else
		s_matmul[tid] = 0;
	if ((start + blockDim.x + tid) < cols)
		s_matmul[tid + blockDim.x] = weights[rowId * cols + start + blockDim.x + tid] * input[start + blockDim.x + tid];
	else
		s_matmul[tid + blockDim.x] = 0;

	cuda_syncthreads();

	for (int stride = blockDim.x; stride >= 1; stride /= 2) {
		if (tid <= stride) {
			s_matmul[tid] += s_matmul[tid + stride];
		}
		cuda_syncthreads();
	}

	if (tid == 0) {
		output[rowId] = s_matmul[0];
	}*/
}

template <unsigned int block_size>
__global__ void d_multiply_weights_fast1(float *input, float *weights, float *output, int cols, int rows, int batch_size) {
	//__shared__ float s_weights[block_size * 2 * MAT_ROWS_PER_THREAD];
	__shared__ float s_input[block_size * 2 * INPUT_CHUNK_SIZE];

	__shared__ float s_matmul[block_size * MAT_ROWS_PER_THREAD * INPUT_CHUNK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = 2 * block_size * blockIdx.x + tid;

	unsigned int start_rid = blockIdx.y * MAT_ROWS_PER_THREAD;
	unsigned int out_start_id = blockIdx.z * INPUT_CHUNK_SIZE;

	unsigned int in_off = out_start_id * cols;
	unsigned int s_in_off = 2 * block_size * threadIdx.z;

	float w0 = 0;
	float w1 = 0;

	/*for (unsigned int row = 0; row < MAT_ROWS_PER_BLOCK; row++) {
	int rid = row + start_rid;
	if (rid < rows)
	s_matrix[BLOCK_SIZE * row + tid] = weights[rid * cols + i];
	else
	s_matrix[MAT_ROWS_PER_BLOCK * row + tid] = 0;
	}*/


	//Load weights into local memory
	int w_row = 2 * block_size * threadIdx.z;

	if (start_rid + w_row < rows) {
		if (i + block_size < cols) {
			w0 = weights[(start_rid + w_row) * cols + i];
			w1 = weights[(start_rid + w_row) * cols + i + block_size];
		}
		else if (i < cols) {
			w0 = weights[(start_rid + w_row) * cols + i];
		}
	}

	//Load input data into shared memory
	for (unsigned int load = 0; load < WEIGHT_LOAD_SIZE; load++) {
		unsigned int s_load_offset = load * 2 * block_size * MAT_ROWS_PER_THREAD;
		unsigned int load_offset = load * cols * MAT_ROWS_PER_THREAD;
		if (i + block_size < cols) {
			s_input[s_in_off + s_load_offset + tid + block_size] = input[i + block_size + in_off + load_offset];
			s_input[s_in_off + s_load_offset + tid] = input[i + in_off + load_offset];
		}
		else if (i < cols) {
			s_input[s_in_off + s_load_offset + tid + block_size] = 0;
			s_input[s_in_off + s_load_offset + tid] = input[i + in_off + load_offset];
		}
		else {
			s_input[s_in_off + s_load_offset + tid + block_size] = 0;
			s_input[s_in_off + s_load_offset + tid] = 0;
		}
	}

	cuda_syncthreads();

	if (i + block_size < cols) {
		for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
			unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
			unsigned int in_row = row * 2 * block_size;
			s_matmul[tid + matmul_row * block_size] = w1 * s_input[in_row + tid + block_size] + w0 * s_input[in_row + tid];
		}
	}
	else if (i < cols) {
		for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
			unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
			unsigned int in_row = row * 2 * block_size;
			s_matmul[tid + matmul_row * block_size] = w0 * s_input[in_row + tid];
		}
	}
	else {
		for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
			unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
			s_matmul[tid + matmul_row * block_size] = 0;
		}
	}

	cuda_syncthreads();

	if (block_size >= 512) {
		if (tid < 256) {
			for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
				unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
				s_matmul[tid + matmul_row * block_size] = s_matmul[tid + matmul_row * block_size + 256];
			}
			cuda_syncthreads();
		}
	}
	if (block_size >= 256) {
		if (tid < 128) {
			for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
				unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
				s_matmul[tid + matmul_row * block_size] = s_matmul[tid + matmul_row * block_size + 128];
			}
			cuda_syncthreads();
		}
	}
	if (block_size >= 128) {
		if (tid < 64) {
			for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
				unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
				s_matmul[tid + matmul_row * block_size] = s_matmul[tid + matmul_row * block_size + 64];
			}
			cuda_syncthreads();
		}
	}

	if (tid < 32) {
		for (unsigned int row = 0; row < INPUT_CHUNK_SIZE; row++) {
			unsigned int matmul_row = row * MAT_ROWS_PER_THREAD + threadIdx.z;
			warp_reduce<block_size>(&s_matmul[matmul_row * block_size], tid);
		}
	}

	if (tid == 0) {
		if (blockIdx.x == 0) {
			for (unsigned int in = 0; in < INPUT_CHUNK_SIZE; in++) {
				unsigned int out_id = out_start_id + in;
				output[start_rid + threadIdx.z + out_id * rows] = s_matmul[0];
			}
		}
		else {
			for (unsigned int in = 0; in < INPUT_CHUNK_SIZE; in++) {
				unsigned int out_id = out_start_id + in;
				output[start_rid + threadIdx.z + out_id * rows] += s_matmul[0];
			}
		}
	}
}

template <unsigned int block_size>
__global__ void d_multiply_weights_fast(float *input, float *weights, float *output, int cols, int rows) {
	__shared__ float s_input[block_size * 2];

	__shared__ float s_matmul[block_size * MAT_ROWS_PER_THREAD];

	unsigned int tid = threadIdx.x;
	unsigned int i = 2 * block_size * blockIdx.x + tid;

	unsigned int start_rid = blockIdx.y * MAT_ROWS_PER_THREAD;
	unsigned int out_id = blockIdx.z;

	unsigned int inOff = out_id * cols;

	/*for (unsigned int row = 0; row < MAT_ROWS_PER_BLOCK; row++) {
	int rid = row + start_rid;
	if (rid < rows)
	s_matrix[BLOCK_SIZE * row + tid] = weights[rid * cols + i];
	else
	s_matrix[MAT_ROWS_PER_BLOCK * row + tid] = 0;
	}*/

	if (i + block_size < cols) {
		s_input[tid + block_size] = input[i + block_size + inOff];
		s_input[tid] = input[i + inOff];
	}
	else if (i < cols) {
		s_input[tid + block_size] = 0;
		s_input[tid] = input[i + inOff];
	}
	else {
		s_input[tid + block_size] = 0;
		s_input[tid] = 0;
	}

	cuda_syncthreads();

	if (i + block_size < cols) {
		for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
			unsigned int rid = row + start_rid;
			s_matmul[tid + row * block_size] = weights[rid * cols + i + block_size] * s_input[tid + block_size] + weights[rid * cols + i] * s_input[tid];
		}
	}
	else if (i < cols) {
		for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
			unsigned int rid = row + start_rid;
			s_matmul[tid + row * block_size] = weights[rid * cols + i] * s_input[tid];
		}
	}
	else {
		for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
			unsigned int rid = row + start_rid;
			s_matmul[tid + row * block_size] = 0;
		}
	}

	cuda_syncthreads();

	if (block_size >= 512) {
		if (tid < 256) {
			for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
				s_matmul[tid + row * block_size] += s_matmul[tid + row * block_size + 256];
			}
			cuda_syncthreads();
		}
	}
	if (block_size >= 256) {
		if (tid < 128) {
			for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
				s_matmul[tid + row * block_size] += s_matmul[tid + row * block_size + 128];
			}
			cuda_syncthreads();
		}
	}
	if (block_size >= 128) {
		if (tid < 64) {
			for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
				s_matmul[tid + row * block_size] += s_matmul[tid + row * block_size + 64];
			}
			cuda_syncthreads();
		}
	}

	if (tid < 32) {
		for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
			warp_reduce<block_size>(&s_matmul[row * block_size], tid);
		}
	}

	if (tid == 0) {
		if (blockIdx.x == 0)
			for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
				output[start_rid + row + out_id * rows] = s_matmul[0];
			}
		else
			for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
				output[start_rid + row + out_id * rows] += s_matmul[0];
			}
	}
}

__global__ void d_multiply_matrix_vector(float *input, float *weights, float *output, 
	unsigned int cols, unsigned int rows, unsigned int stagger_size, unsigned int n_inputs) {
	//declare shared memory for current block
	__shared__ float s_input[MATMUL_BLOCK_SIZEx2 * INPUT_CHUNK_SIZE];
	__shared__ float s_weights[MATMUL_BLOCK_SIZEx2 * MAT_ROWS_PER_THREAD];
	__shared__ float s_matmul[MATMUL_BLOCK_SIZE * INPUT_CHUNK_SIZE * MAT_ROWS_PER_THREAD];

	//declare thread indices etc

	//block dims: {caluclation size, 1, input size}
	//grid dims: {total cols / calc size, total rows / rows per thread, total inputs / input size per block}

	unsigned int tid = threadIdx.x;
	unsigned int g_tid = MATMUL_BLOCK_SIZEx2 * blockIdx.x + tid;
	unsigned int inid = threadIdx.z;
	unsigned int g_inid = INPUT_CHUNK_SIZE * blockIdx.z + inid;
	unsigned int g_rowid = blockIdx.y * MAT_ROWS_PER_THREAD;

	unsigned int input0_id = g_inid * stagger_size + g_tid;
	unsigned int input1_id = input0_id + MATMUL_BLOCK_SIZE;

	unsigned int s_input0_id = inid * MATMUL_BLOCK_SIZEx2 + tid;
	unsigned int s_input1_id = s_input0_id + MATMUL_BLOCK_SIZE;

	unsigned int inid_x_mrpt = inid * MAT_ROWS_PER_THREAD;

	if (g_inid >= n_inputs)
		return;
	
	//load inputs into shared memory

	if (g_inid < n_inputs) {
		if (g_tid + MATMUL_BLOCK_SIZE < cols) {
			s_input[s_input0_id] = input[input0_id];
			s_input[s_input1_id] = input[input1_id];
		}
		else if (g_tid < cols) {
			s_input[s_input0_id] = input[input0_id];
			s_input[s_input1_id] = 0;
		}
		else {
			s_input[s_input0_id] = 0;
			s_input[s_input1_id] = 0;
		}
	}
	else {
		s_input[s_input0_id] = 0;
		s_input[s_input1_id] = 0;
	}

	//load weights into shared memory reusing same threads

	if (g_tid + MATMUL_BLOCK_SIZE < cols) {
		for (unsigned int load = 0; load < WEIGHT_LOAD_SIZE * INPUT_CHUNK_SIZE; load += INPUT_CHUNK_SIZE) {
			unsigned int load_id = load + inid;
			unsigned int g_load_id = load_id + g_rowid;

			unsigned int row_load_id = load_id * MATMUL_BLOCK_SIZEx2 + tid;
			unsigned int g_row_load_id = (g_load_id) * cols + g_tid;

			if (load_id < MAT_ROWS_PER_THREAD && g_load_id < rows) {
				s_weights[row_load_id + MATMUL_BLOCK_SIZE] = weights[g_row_load_id + MATMUL_BLOCK_SIZE];
				s_weights[row_load_id] = weights[g_row_load_id];
			}
		}
	}
	else if (g_tid < cols) {
		for (unsigned int load = 0; load < WEIGHT_LOAD_SIZE * INPUT_CHUNK_SIZE; load += INPUT_CHUNK_SIZE) {
			unsigned int load_id = load + inid;
			unsigned int g_load_id = load_id + g_rowid;

			unsigned int row_load_id = load_id * MATMUL_BLOCK_SIZEx2 + tid;
			unsigned int g_row_load_id = (g_load_id) * cols + g_tid;

			if (load_id < MAT_ROWS_PER_THREAD && g_load_id < rows) {
				s_weights[row_load_id + MATMUL_BLOCK_SIZE] = 0;
				s_weights[row_load_id] = weights[g_row_load_id];
			}
		}
	}
	else {
		for (unsigned int load = 0; load < WEIGHT_LOAD_SIZE * INPUT_CHUNK_SIZE; load += INPUT_CHUNK_SIZE) {
			unsigned int load_id = load + inid;
			unsigned int g_load_id = load_id + g_rowid;

			unsigned int row_load_id = load_id * MATMUL_BLOCK_SIZEx2 + tid;
			unsigned int g_row_load_id = (g_load_id)* cols + g_tid;

			if (load_id < MAT_ROWS_PER_THREAD && g_load_id < rows) {
				s_weights[row_load_id + MATMUL_BLOCK_SIZE] = 0;
				s_weights[row_load_id] = 0;
			}
		}
	}

	cuda_syncthreads();

	//compute multiplication elements

	for (unsigned int m_row = 0; m_row < MAT_ROWS_PER_THREAD; m_row++) {
		unsigned int matmul_row = inid_x_mrpt + m_row;
		unsigned int weight_id = m_row * MATMUL_BLOCK_SIZEx2 + tid;

		s_matmul[matmul_row * MATMUL_BLOCK_SIZE + tid] = s_weights[weight_id + MATMUL_BLOCK_SIZE] *
			s_input[s_input1_id] +
			s_weights[weight_id] *
			s_input[s_input0_id];
	}

	cuda_syncthreads();

	//use parallel reduction to calculate partial sums

	for (unsigned int row = 0; row < MAT_ROWS_PER_THREAD; row++) {
		unsigned int matmul_index = (inid_x_mrpt + row) * MATMUL_BLOCK_SIZE + tid;
		if (MATMUL_BLOCK_SIZE >= 512) {
			if (tid < 256) {
				s_matmul[matmul_index] += s_matmul[matmul_index + 256];
			}
			cuda_syncthreads();
		}
		if (MATMUL_BLOCK_SIZE >= 256) {
			if (tid < 128) {
				s_matmul[matmul_index] += s_matmul[matmul_index + 128];
			}
			cuda_syncthreads();
		}
		if (MATMUL_BLOCK_SIZE >= 128) {
			if (tid < 64) {
				s_matmul[matmul_index] += s_matmul[matmul_index + 64];
			}
			cuda_syncthreads();
		}
		if (tid < 32)
			warp_reduce_to_zero<MATMUL_BLOCK_SIZE>(&s_matmul[matmul_index]);
	}

	//cuda_syncthreads();

	unsigned int g_row_index = g_inid * rows + g_rowid;

	//use block size / rows per thread as for loop (just in case more rows than block chunk size)
	//use these threads as each row provided tid < rows per thread
	//no need for iteration (unless huge no. of rows per thread)

	for (int output_load = 0; output_load < MAT_ROWS_PER_THREAD * OUTPUTS_PER_THREAD; output_load += MAT_ROWS_PER_THREAD) {
		unsigned int row = g_rowid + output_load + tid;
		if (tid < MAT_ROWS_PER_THREAD && row < rows) {
			unsigned int g_outid = (g_inid * rows + row) * gridDim.x + blockIdx.x;
			unsigned int matmul_index = (inid_x_mrpt + output_load + tid) * MATMUL_BLOCK_SIZE;
			output[g_outid] = s_matmul[matmul_index];
		}
	}
}

__global__ void d_multiply_matrices(float * A, float * B, float * output, unsigned int A_rows, unsigned int A_cols, unsigned int B_rows,
	unsigned int B_cols) {
	//declare shared memory for current block
	//__shared__ float s_A[M_TILE_SIZE_X * M_TILE_SIZE_X];
	//__shared__ float s_B[M_TILE_SIZE_X * M_TILE_SIZE_X];
	//__shared__ float s_matmul[M_TILE_SIZE_X * M_TILE_SIZE_X];
	__shared__ float s_A[M_TILE_SIZE_X * M_TILE_SIZE_Y];
	__shared__ float s_B[M_TILE_SIZE_X * M_TILE_SIZE_Y];
	__shared__ float s_matmul[M_TILE_SIZE_Y * M_TILE_SIZE_Y];

	unsigned int col_l = threadIdx.x;
	unsigned int row_l = threadIdx.y;

	/*unsigned int A_col_g = blockIdx.x * M_TILE_SIZE_X + col_l;
	unsigned int A_row_g = blockIdx.y * M_TILE_SIZE_X + row_l;

	unsigned int B_col_g = blockIdx.z * M_TILE_SIZE_X + col_l;
	unsigned int B_row_g = blockIdx.x * M_TILE_SIZE_X + row_l;

	unsigned int tile_k_size = A_cols - blockIdx.x * M_TILE_SIZE_X;*/

	unsigned int A_col_g = blockIdx.x * M_TILE_SIZE_X + col_l;
	unsigned int A_row_g = blockIdx.y * M_TILE_SIZE_Y + row_l;

	unsigned int B_col_g = blockIdx.z * M_TILE_SIZE_Y + col_l;
	unsigned int B_row_g = blockIdx.x * M_TILE_SIZE_X + row_l;

	/*unsigned int tile_k_size = A_cols - blockIdx.x * M_TILE_SIZE_X;

	if (tile_k_size > M_TILE_SIZE_X)
		tile_k_size = M_TILE_SIZE_X;*/

	unsigned int tile_k_size = A_cols - blockIdx.x * M_TILE_SIZE_X;

	if (tile_k_size > M_TILE_SIZE_X)
		tile_k_size = M_TILE_SIZE_X;

	unsigned int a_index_l = row_l * M_TILE_SIZE_X + col_l;
	unsigned int b_index_l = col_l * M_TILE_SIZE_Y + row_l;

	if (A_row_g < A_rows && A_col_g < A_cols) {
		s_A[a_index_l] = A[A_row_g * A_cols + A_col_g];
		//s_A[index_l] = A[tid];
	}
	else {
		s_A[a_index_l] = 0;
	}
	if (B_row_g < B_rows && B_col_g < B_cols) {
		//s_B[index_l] = B[B_row_g * B_cols + B_col_g];
		s_B[b_index_l] = B[B_col_g * B_rows + B_row_g];
		//s_B[index_l] = B[tid];
	}
	else {
		s_B[b_index_l] = 0;
	}
	s_matmul[a_index_l] = 0;

	cuda_syncthreads();

	if (col_l > M_TILE_SIZE_Y)
		return;

	for (int k = 0; k < tile_k_size; k++) {
		//s_matmul[index_l] = s_A[row_l * M_TILE_SIZE_X + k] * s_B[k * M_TILE_SIZE_Y + col_l];
		s_matmul[b_index_l] = s_A[row_l * M_TILE_SIZE_X + k] * s_B[col_l * M_TILE_SIZE_Y + k];
	}
	
	cuda_syncthreads();

	//warp_reduce_to_zero<M_TILE_SIZE>(&s_matmul[mm_index]);

	if (A_row_g < A_rows && B_col_g < B_cols)
		output[gridDim.x * (B_col_g * A_rows + A_row_g) + blockIdx.x] = s_matmul[b_index_l];
	//output[gridDim.x * (A_row_g * B_cols + B_col_g) + blockIdx.x] = s_matmul[index_l];
}

template <unsigned int reduce_size>
__global__ void d_compute_sums(float * partial_sums, float * output, int partial_sum_count, int elem_count) {
	__shared__ float s_sums[2048];

	unsigned int psum_index = threadIdx.y;
	unsigned int element_index = (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int element_location = element_index * partial_sum_count + psum_index;

	unsigned int s_i = reduce_size * threadIdx.x;
	unsigned int ps_i0 = s_i + psum_index;
	unsigned int ps_i1 = ps_i0 + reduce_size / 2;

	if (element_index > elem_count)
		return;

	if (psum_index + reduce_size / 2 < partial_sum_count) {
		s_sums[ps_i0] = partial_sums[element_location];
		s_sums[ps_i1] = partial_sums[element_location + reduce_size / 2];
	}
	else if (psum_index < partial_sum_count) {
		s_sums[ps_i0] = partial_sums[element_location];
		s_sums[ps_i1] = 0;
	}
	else {
		s_sums[ps_i0] = 0;
		s_sums[ps_i1] = 0;
	}

	cuda_syncthreads();

	if (reduce_size >= 2048) {
		if (psum_index < 1024) {
			s_sums[ps_i0] += s_sums[ps_i0 + 1024];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 1024) {
		if (psum_index < 512) {
			s_sums[ps_i0] += s_sums[ps_i0 + 512];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 512) {
		if (psum_index < 256) {
			s_sums[ps_i0] += s_sums[ps_i0 + 256];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 256) {
		if (psum_index < 128) {
			s_sums[ps_i0] += s_sums[ps_i0 + 128];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 128) {
		if (psum_index < 64) {
			s_sums[ps_i0] += s_sums[ps_i0 + 64];
		}
		cuda_syncthreads();
	}

	if (reduce_size >= 64) {
		if (psum_index < 32) {
			s_sums[ps_i0] += s_sums[ps_i0 + 32];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 32) {
		if (psum_index < 16) {
			s_sums[ps_i0] += s_sums[ps_i0 + 16];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 16) {
		if (psum_index < 8) {
			s_sums[ps_i0] += s_sums[ps_i0 + 8];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 8) {
		if (psum_index < 4) {
			s_sums[ps_i0] += s_sums[ps_i0 + 4];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 4) {
		if (psum_index < 2) {
			s_sums[ps_i0] += s_sums[ps_i0 + 2];
		}
		cuda_syncthreads();
	}
	if (reduce_size >= 2) {
		if (psum_index < 1) {
			s_sums[ps_i0] += s_sums[ps_i0 + 1];
		}
		cuda_syncthreads();
	}

	if (psum_index == 0) {
		output[element_index] = s_sums[s_i];
	}
}

template <unsigned int reduce_block_size>
__global__ void d_reduce_matvec_rows(float * partial_sums, float * output, unsigned int rows, unsigned int reduce_size, unsigned int stagger_size) {
	__shared__ float s_reduce[reduce_block_size * MAT_ROWS_PER_THREAD];

	unsigned int rid = threadIdx.x;
	unsigned int rowid = threadIdx.y;
	unsigned int g_rowid = MAT_ROWS_PER_THREAD * blockIdx.y + rowid;
	unsigned int g_inid = blockIdx.x;

	int x = blockDim.x + gridDim.x;

	unsigned int l_row = rowid * reduce_block_size;
	unsigned int reduce_index = l_row + rid;

	unsigned int p_s_index = g_inid * rows * reduce_size + rid;

	if (g_rowid < rows) {
		if (rid + reduce_block_size < reduce_size) {
			s_reduce[reduce_index] = partial_sums[p_s_index] + partial_sums[p_s_index + reduce_block_size];
		}
		else if (rid < reduce_size) {
			s_reduce[reduce_index] = partial_sums[p_s_index];
		}
		else {
			s_reduce[reduce_index] = 0;
		}
	}
	else {
		s_reduce[reduce_index] = 0;
	}

	cuda_syncthreads();

	if (reduce_block_size >= 512) {
		if (rid < 256) {
			s_reduce[reduce_index] += s_reduce[reduce_index + 256];
		}
		cuda_syncthreads();
	}
	if (reduce_block_size >= 256) {
		if (rid < 128) {
			s_reduce[reduce_index] += s_reduce[reduce_index + 128];
		}
		cuda_syncthreads();
	}
	if (reduce_block_size >= 128) {
		if (rid < 64) {
			s_reduce[reduce_index] += s_reduce[reduce_index + 64];
		}
		cuda_syncthreads();
	}

	if (rid < 32) {
		warp_reduce_to_zero<reduce_block_size>(&s_reduce[reduce_index]);
	}

	if (rid == 0 && g_rowid < rows) {
		output[g_inid * stagger_size + g_rowid] = s_reduce[0];// l_row];
	}
}

template <unsigned int block_size>
__global__ void d_multiply_weights_naive(float *input, float *weights, float *output, int cols, int rows) {
	__shared__ float s_input[block_size];
	__shared__ float s_weights[block_size];
	__shared__ float s_row[block_size];

	int tid = threadIdx.x;
	int i = block_size * blockIdx.x + tid;
	int row = blockIdx.y;
	int n = blockIdx.z;

	s_input[tid] = input[n * rows + i];
	s_weights[tid] = weights[row * cols + i];

	cuda_syncthreads();

	s_row[tid] = s_input[tid] * s_weights[tid];

	cuda_syncthreads();

	for (int s = block_size / 2; s > 0; s >>= 1) {
		if (tid < s)
			s_row[tid] = s_row[tid + s];
		cuda_syncthreads();
	}

	if (tid == 0)
		output[n * rows + row] = s_row[0];

	/*float val = 0;
	for (int i = 0; i < rows; i++) {
		val += weights[row * cols + i] * input[n * rows + i];
	}

	output[n * rows + row] = val;*/
}

__global__ void d_apply_relu(float *input, float *output, int size, float alpha) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size) {
		if (input[id] > 0)
			output[id] = input[id];
		else
			output[id] = alpha * input[id];
	}
}

template <unsigned int block_size>
__global__ void d_apply_softmax(float *input, float *output, int input_size, float beta) {
	__shared__ float s_sum[block_size * 2], s_exp_vals[block_size * 2];

	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int inid = blockIdx.y;
	
	unsigned int exp_index = inid * input_size + tid;

	float tmp0 = 0;
	float tmp1 = 0;

	if (tid + block_size < input_size) {
		tmp0 = expf(beta * input[exp_index + block_size]);
		tmp1 = expf(beta * input[exp_index]);
	}

	else if (tid < input_size) {
		tmp0 = 0;
		tmp1 = expf(beta * input[exp_index]);
	}
	else {
		tmp0 = 0;
		tmp1 = 0;
	}

	/*if (isinf(tmp0))
		tmp0 = 4294967296 * 4294967296;
	if (isinf(tmp1))
		tmp1 = 4294967296 * 4294967296;*/

	s_sum[tid + block_size] = tmp0;
	s_sum[tid] = tmp1;
	s_exp_vals[tid + block_size] = tmp0;
	s_exp_vals[tid] = tmp1;

	cuda_syncthreads();

	if (block_size >= 1024) {
		if (tid < 1024) {
			s_sum[tid] += s_sum[tid + 1024];
		}
		cuda_syncthreads();
	}
	if (block_size >= 512) {
		if (tid < 512) {
			s_sum[tid] += s_sum[tid + 512];
		}
		cuda_syncthreads();
	}
	if (block_size >= 256) {
		if (tid < 256) {
			s_sum[tid] += s_sum[tid + 256];
		}
		cuda_syncthreads();
	}
	if (block_size >= 128) {
		if (tid < 128) {
			s_sum[tid] += s_sum[tid + 128];
		}
		cuda_syncthreads();
	}
	if (block_size >= 64) {
		if (tid < 64) {
			s_sum[tid] += s_sum[tid + 64];
		}
		cuda_syncthreads();
	}
	if (block_size >= 32) {
		if (tid < 32) {
			s_sum[tid] += s_sum[tid + 32];
		}
		cuda_syncthreads();
	}
	if (block_size >= 16) {
		if (tid < 16) {
			s_sum[tid] += s_sum[tid + 16];
		}
		cuda_syncthreads();
	}
	if (block_size >= 8) {
		if (tid < 8) {
			s_sum[tid] += s_sum[tid + 8];
		}
		cuda_syncthreads();
	}
	if (block_size >= 4) {
		if (tid < 4) {
			s_sum[tid] += s_sum[tid + 4];
		}
		cuda_syncthreads();
	}
	if (block_size >= 2) {
		if (tid < 2) {
			s_sum[tid] += s_sum[tid + 2];
		}
		cuda_syncthreads();
	}
	if (block_size >= 1) {
		if (tid < 1) {
			s_sum[tid] += s_sum[tid + 1];
		}
		cuda_syncthreads();
	}

	if (tid + block_size < input_size) {
		output[exp_index + block_size] = s_exp_vals[tid + block_size] / s_sum[0];
		output[exp_index] = s_exp_vals[tid] / s_sum[0];
	}
	else if (tid < input_size) {
		output[exp_index] = s_exp_vals[tid] / s_sum[0];
	}
}

__global__ void d_relu_derivative(float * input, float * output, int size, float alpha) {
	unsigned int tid = threadIdx.x + BLOCK_SIZE * blockIdx.x;

	if (tid < size) {
		if (input[tid] > 0)
			output[tid] = 1;
		else
			output[tid] = alpha;
	}
}

template <unsigned int block_size>
__global__ void d_batch_norm(float * input, float * output, int size, int num) {
	__shared__ float s_mean[block_size * 2];
}

__global__ void d_batch_norm_derivative(float * input, float * output, int size, int num) {

}

/*__global__ void d_softmax_derivative(float * input, float * output, int size, float beta) {
	unsigned int tid = threadIdx.x + BLOCK_SIZE * blockIdx.x;

	if (tid < size) {
		float s = input[tid];
		output[tid] = beta * s * (1 - s);
	}
}*/

__global__ void d_hadamard_product(float * a, float * b, float * output, int size) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < size) {
		output[tid] = a[tid] * b[tid];
	}
}

__global__ void d_transpose(float * input, float * output, int rows, int cols) {
	unsigned int colid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int rowid = threadIdx.y + blockIdx.y * blockDim.y;

	if (colid < cols && rowid < rows) {
		output[colid * rows + rowid] = input[rowid * cols + colid];
	}
}

template <unsigned int block_size>
__global__ void d_average_vector(float * input, float * output, int input_size, int num, int divisor) {
	//call with block_size / 2 threads in x
	//call with average_chunk_size threads in y

	unsigned int tid = threadIdx.x;
	unsigned int g_inid = blockIdx.y;

	__shared__ float s_sums[block_size * 2];
	
	//for (int i = tid + blockDim.x * inid; i < BLOCK_SIZE * AVERAGE_CHUNK_SIZE; i += blockDim.x * blockDim.y)
	//	s_sums[i] = 0;
	//cuda_syncthreads();

	if (tid + block_size < num) {
		s_sums[tid + block_size] = input[(tid + block_size) * input_size + g_inid];
		s_sums[tid] = input[tid * input_size + g_inid];
	}
	else if (tid < num) {
		s_sums[tid + block_size] = 0;
		s_sums[tid] = input[tid * num + g_inid];
	}
	else {
		s_sums[tid + block_size] = 0;
		s_sums[tid] = 0;
	}

	cuda_syncthreads();

	if (block_size >= 1024) {
		if (tid < 1024) {
			s_sums[tid] += s_sums[tid + 1024];
		}
		cuda_syncthreads();
	}
	if (block_size >= 512) {
		if (tid < 512) {
			s_sums[tid] += s_sums[tid + 512];
		}
		cuda_syncthreads();
	}
	if (block_size >= 256) {
		if (tid < 256) {
			s_sums[tid] += s_sums[tid + 256];
		}
		cuda_syncthreads();
	}
	if (block_size >= 128) {
		if (tid < 128) {
			s_sums[tid] += s_sums[tid + 128];
		}
		cuda_syncthreads();
	}
	if (block_size >= 64) {
		if (tid < 64) {
			s_sums[tid] += s_sums[tid + 64];
		}
		cuda_syncthreads();
	}
	if (block_size >= 32) {
		if (tid < 32) {
			s_sums[tid] += s_sums[tid + 32];
		}
		cuda_syncthreads();
	}
	if (block_size >= 16) {
		if (tid < 16) {
			s_sums[tid] += s_sums[tid + 16];
		}
		cuda_syncthreads();
	}
	if (block_size >= 8) {
		if (tid < 8) {
			s_sums[tid] += s_sums[tid + 8];
		}
		cuda_syncthreads();
	}
	if (block_size >= 4) {
		if (tid < 4) {
			s_sums[tid] += s_sums[tid + 4];
		}
		cuda_syncthreads();
	}
	if (block_size >= 2) {
		if (tid < 2) {
			s_sums[tid] += s_sums[tid + 2];
		}
		cuda_syncthreads();
	}
	if (block_size >= 1) {
		if (tid < 1) {
			s_sums[tid] += s_sums[tid + 1];
		}
		cuda_syncthreads();
	}
	//if (tid < 32)
	//	warp_reduce_to_zero<AVERAGE_CHUNK_SIZE>(&s_sums[index]);

	if (tid == 0) {
		output[g_inid] = s_sums[0] / divisor;
		/*if (g_tid + block_size < input_size) {
			output[tid + block_size] = s_sums[tid + block_size] / divisor;
			output[tid] = s_sums[tid] / divisor;
		}
		else if (g_tid < input_size) {
			output[tid] = s_sums[tid] / divisor;
		}*/
	}
}

template <typename T_i, typename T_o>
__global__ void d_scalar_matrix_mul(T_i * input, T_o * output, float scalar, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < size) {
		output[tid] = input[tid] * scalar;
	}
}

template <unsigned int block_size>
__global__ void d_average_value(float * input, float * output, int size, float divisor) {
	__shared__ float s_sum[block_size * 2];

	int id0 = threadIdx.x;
	int id1 = id0 + block_size;

	if (id1 < size) {
		s_sum[id1] = input[id1];
		s_sum[id0] = input[id0];
	}
	else if (id0 < size) {
		s_sum[id1] = 0;
		s_sum[id0] = input[id0];
	}
	else {
		s_sum[id1] = 0;
		s_sum[id0] = 0;
	}

	cuda_syncthreads();

	if (block_size >= 1024) {
		if (id0 < 1024) {
			s_sum[id0] += s_sum[id0 + 1024];
		}
		cuda_syncthreads();
	}
	if (block_size >= 512) {
		if (id0 < 512) {
			s_sum[id0] += s_sum[id0 + 512];
		}
		cuda_syncthreads();
	}
	if (block_size >= 256) {
		if (id0 < 256) {
			s_sum[id0] += s_sum[id0 + 256];
		}
		cuda_syncthreads();
	}
	if (block_size >= 128) {
		if (id0 < 128) {
			s_sum[id0] += s_sum[id0 + 128];
		}
		cuda_syncthreads();
	}
	if (block_size >= 64) {
		if (id0 < 64) {
			s_sum[id0] += s_sum[id0 + 64];
		}
		cuda_syncthreads();
	}
	if (block_size >= 32) {
		if (id0 < 32) {
			s_sum[id0] += s_sum[id0 + 32];
		}
		cuda_syncthreads();
	}
	if (block_size >= 16) {
		if (id0 < 16) {
			s_sum[id0] += s_sum[id0 + 16];
		}
		cuda_syncthreads();
	}
	if (block_size >= 8) {
		if (id0 < 8) {
			s_sum[id0] += s_sum[id0 + 8];
		}
		cuda_syncthreads();
	}
	if (block_size >= 4) {
		if (id0 < 4) {
			s_sum[id0] += s_sum[id0 + 4];
		}
		cuda_syncthreads();
	}
	if (block_size >= 2) {
		if (id0 < 2) {
			s_sum[id0] += s_sum[id0 + 2];
		}
		cuda_syncthreads();
	}
	if (block_size >= 1) {
		if (id0 < 1) {
			s_sum[id0] += s_sum[id0 + 1];
		}
		cuda_syncthreads();
	}

	if (id0 == 0)
		output[0] = (float)s_sum[0] / divisor;
}

__global__ void d_sub_pds(float * d_matrix, float * d_derivatives, float * d_output, int size, float learning_rate) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < size)
		d_output[id] = d_matrix[id] - d_derivatives[id] * learning_rate;
}

/*__global__ void d_distributive_hadamard_transpose(float * in, float * output, float * matrix, int rows, int cols, unsigned int n_inputs) {
	//THIS IS JUST STANDARD MATRIX MULTIPLICATION BETWEEN THE TRANSPOSE AND THE VECTOR.

	__shared__ float s_mat[MATMUL_BLOCK_SIZEx2 * MAT_ROWS_PER_THREAD];

	unsigned int tid = threadIdx.x;
	unsigned int g_tid = MATMUL_BLOCK_SIZEx2 * blockIdx.x + tid;

	for (unsigned int load = 0; load < WEIGHT_LOAD_SIZE * INPUT_CHUNK_SIZE; load += INPUT_CHUNK_SIZE) {
		
	}

	//call with blocksize threads in x, 1 in y, and input block size in z
	//ceil div blocksize blocks in x, ceil div rows in y, ceil div block size in z

	//load in (transpose?) matrix into s_mat -> load sequentially from global mem into correct transpose in shared mem
	//sum up columns
	//multiply by input
	//write to partial output
}*/

template <unsigned int block_size>
__global__ void d_squared_error(float *input, float *target, float *output, int size) {
	extern __shared__ float s_cost[];

	int tid = threadIdx.x;
	int i = 2 * block_size * blockIdx.x + tid;

	float diff[2];
	diff[0] = input[i] - target[i];
	diff[1] = input[i + block_size] - target[i + block_size];
	s_cost[tid] += diff[0] * diff[0] + diff[1] * diff[1];

	cuda_syncthreads();

	if (block_size >= 512) {
		if (tid < 256) {
			s_cost[tid] += s_cost[tid + 256];
		}
		cuda_syncthreads();
	}
	if (block_size >= 256) {
		if (tid < 128) {
			s_cost[tid] += s_cost[tid + 128];
		}
		cuda_syncthreads();
	}
	if (block_size >= 128) {
		if (tid < 64) {
			s_cost[tid] += s_cost[tid + 64];
		}
		cuda_syncthreads();
	}

	if (tid < 32) {
		warp_reduce<block_size>(s_cost, tid);
	}

	if (tid == 0)
		output[0] += s_cost[0];
}

__global__ void d_squared_error_derivative(float * input, float * target, float * output, int size) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (tid < size)
		output[tid] = 2 * (input[tid] - target[tid]);
}

template <unsigned int block_size>
__global__ void d_softmax_cross_entropy(float *input, float *target, float *output, int size, int num) {
	__shared__ double s_exps[block_size * 2];
	__shared__ float s_cost[block_size * 2];

	unsigned int x_id0 = threadIdx.x;
	unsigned int x_id1 = x_id0 + block_size;
	//unsigned int in_id = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int in_id = blockIdx.y;
	unsigned int in_index0 = in_id * size + x_id0;
	unsigned int in_index1 = in_id * size + x_id1;

	//if (in_id > num)
	//	return;

	if (x_id1 < size) {
		float tmp0 = input[in_index0];
		float tmp1 = input[in_index1];
		s_exps[x_id0] = exp(tmp0);
		s_exps[x_id1] = exp(tmp1);
		s_cost[x_id0] = target[in_index0] * tmp0;
		s_cost[x_id1] = target[in_index1] * tmp1;
	}
	else if (x_id0 < size) {
		float tmp0 = input[in_index0];
		s_exps[x_id0] = exp(tmp0);
		s_exps[x_id1] = 0;
		s_cost[x_id0] = target[in_index0] * tmp0;
		s_cost[x_id1] = 0;
	}
	else {
		s_exps[x_id0] = 0;
		s_exps[x_id1] = 0;
		s_cost[x_id0] = 0;
		s_cost[x_id1] = 0;
	}

	cuda_syncthreads();

	if (block_size >= 1024) {
		if (x_id0 < 1024) {
			s_exps[x_id0] += s_exps[x_id0 + 1024];
			s_cost[x_id0] += s_cost[x_id0 + 1024];
		}
		cuda_syncthreads();
	}
	if (block_size >= 512) {
		if (x_id0 < 512) {
			s_exps[x_id0] += s_exps[x_id0 + 512];
			s_cost[x_id0] += s_cost[x_id0 + 512];
		}
		cuda_syncthreads();
	}
	if (block_size >= 256) {
		if (x_id0 < 256) {
			s_exps[x_id0] += s_exps[x_id0 + 256];
			s_cost[x_id0] += s_cost[x_id0 + 256];
		}
		cuda_syncthreads();
	}
	if (block_size >= 128) {
		if (x_id0 < 128) {
			s_exps[x_id0] += s_exps[x_id0 + 128];
			s_cost[x_id0] += s_cost[x_id0 + 128];
		}
		cuda_syncthreads();
	}
	if (block_size >= 64) {
		if (x_id0 < 64) {
			s_exps[x_id0] += s_exps[x_id0 + 64];
			s_cost[x_id0] += s_cost[x_id0 + 64];
		}
		cuda_syncthreads();
	}
	if (block_size >= 32) {
		if (x_id0 < 32) {
			s_exps[x_id0] += s_exps[x_id0 + 32];
			s_cost[x_id0] += s_cost[x_id0 + 32];
		}
		cuda_syncthreads();
	}
	if (block_size >= 16) {
		if (x_id0 < 16) {
			s_exps[x_id0] += s_exps[x_id0 + 16];
			s_cost[x_id0] += s_cost[x_id0 + 16];
		}
		cuda_syncthreads();
	}
	if (block_size >= 8) {
		if (x_id0 < 8) {
			s_exps[x_id0] += s_exps[x_id0 + 8];
			s_cost[x_id0] += s_cost[x_id0 + 8];
		}
		cuda_syncthreads();
	}
	if (block_size >= 4) {
		if (x_id0 < 4) {
			s_exps[x_id0] += s_exps[x_id0 + 4];
			s_cost[x_id0] += s_cost[x_id0 + 4];
		}
		cuda_syncthreads();
	}
	if (block_size >= 2) {
		if (x_id0 < 2) {
			s_exps[x_id0] += s_exps[x_id0 + 2];
			s_cost[x_id0] += s_cost[x_id0 + 2];
		}
		cuda_syncthreads();
	}
	if (block_size >= 1) {
		if (x_id0 < 1) {
			s_exps[x_id0] += s_exps[x_id0 + 1];
			s_cost[x_id0] += s_cost[x_id0 + 1];
		}
		cuda_syncthreads();
	}

	if (x_id0 == 0 && in_id < num) {
		//output[in_id] = logf(s_exps[0]) - s_cost[0];
		atomic_add(output, logf(s_exps[0]) - s_cost[0]);
	}
}

__global__ void d_softmax_cross_entropy_derivative(float *input, float *target, float *output, int size) {
	unsigned int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int id = x_id + blockIdx.y * size;

	if (x_id < size) {
		output[id] = input[id] - target[id];
	}
}

template <unsigned int block_size>
__device__ void warp_reduce(volatile float *s_pr_array, int tid) {
	if (block_size >= 64) {
		s_pr_array[tid] += s_pr_array[tid + 32];
	}
	if (block_size >= 32) {
		s_pr_array[tid] += s_pr_array[tid + 16];
	}
	if (block_size >= 16) {
		s_pr_array[tid] += s_pr_array[tid + 8];
	}
	if (block_size >= 8) {
		s_pr_array[tid] += s_pr_array[tid + 4];
	}
	if (block_size >= 4) {
		s_pr_array[tid] += s_pr_array[tid + 2];
	}
	if (block_size >= 2) {
		s_pr_array[tid] += s_pr_array[tid + 1];
	}
}

template <unsigned int block_size>
__device__ void warp_reduce_to_zero(volatile float * s_pr_array) {
	if (block_size >= 64)
		s_pr_array[0] += s_pr_array[32];
	if (block_size >= 32)
		s_pr_array[0] += s_pr_array[16];
	if (block_size >= 16)
		s_pr_array[0] += s_pr_array[8];
	if (block_size >= 8)
		s_pr_array[0] += s_pr_array[4];
	if (block_size >= 4)
		s_pr_array[0] += s_pr_array[2];
	if (block_size >= 2)
		s_pr_array[0] += s_pr_array[1];
}

template <unsigned int block_size>
__device__ void warp_reduce_to_zero(volatile double * s_pr_array) {
	if (block_size >= 64)
		s_pr_array[0] += s_pr_array[32];
	if (block_size >= 32)
		s_pr_array[0] += s_pr_array[16];
	if (block_size >= 16)
		s_pr_array[0] += s_pr_array[8];
	if (block_size >= 8)
		s_pr_array[0] += s_pr_array[4];
	if (block_size >= 4)
		s_pr_array[0] += s_pr_array[2];
	if (block_size >= 2)
		s_pr_array[0] += s_pr_array[1];
}

void allocate_device_pointer(float ** d_pointer, int size)
{
	cuda_safe_call(cudaMallocManaged(d_pointer, sizeof(float) * size));
}

void deallocate_device_pointer(float * d_pointer)
{
	if (d_pointer != nullptr)
		cuda_safe_call(cudaFree(d_pointer));
}

void load_data_into_device(float * input_data, float * d_data_p, int size)
{
	cuda_safe_call(cudaMemcpy(d_data_p, input_data, sizeof(float) * size, cudaMemcpyHostToDevice));
}

void retrieve_output_data(float * output_data, float * d_data_p, int size)
{
	cuda_safe_call(cudaMemcpy(output_data, d_data_p, sizeof(float) * size, cudaMemcpyDeviceToHost));
}

void copy_into_device_array(float * input_data, float * d_data_p, int size, int offset)
{
	//cuda_safe_call(cudaMemcpy(&d_data_p[offset], input_data, sizeof(float) * size, cudaMemcpyDeviceToDevice));
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_fast_copy<<<blocks_per_grid, threads_per_block>>>(input_data, &d_data_p[offset], size);
}

void copy_staggered_into_device_array(float * input_data, float * d_data_pointer, int in_dat_size, int out_dat_size, int num)
{
	int threadx = min(in_dat_size, out_dat_size);
	dim3 threads_per_block(threadx, 1);
	dim3 blocks_per_grid(1, num);
	if (threads_per_block.x > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, threadx);
	}
	d_staggered_copy<<<blocks_per_grid, threads_per_block>>>(input_data, d_data_pointer, in_dat_size, out_dat_size);
}

void get_prng(curandGenerator_t * prng, int seed)
{
	curandCreateGenerator(prng, CURAND_RNG_PSEUDO_XORWOW);

	curandSetPseudoRandomGeneratorSeed(*prng, seed);
}

void random_host_array(curandGenerator_t prng, float * array_p, float scale, float offset, int size, int seed)
{
	float *d_array_p;
	cuda_safe_call(cudaMallocManaged(&d_array_p, sizeof(float) * size));

	curandGenerateUniform(prng, d_array_p, size);

	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}

	d_random_scale_offset<<<blocks_per_grid, threads_per_block>>>(d_array_p, scale, offset, size);

	cuda_safe_call(cudaMemcpy(array_p, d_array_p, sizeof(float) * size, cudaMemcpyDeviceToHost));
	cuda_safe_call(cudaFree(d_array_p));
}

/*void fill_array(float * array_p, float value, int size)
{
	float *d_array;
	cuda_safe_call(cudaMallocManaged(&d_array, sizeof(float) * size));

	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_fill_array<<<blocks_per_grid, threads_per_block>>>(d_array, value, size);

	cuda_safe_call(cudaMemcpy(array_p, d_array, sizeof(float) * size, cudaMemcpyDeviceToHost));

	cuda_safe_call(cudaFree(d_array));
}*/

void fill_device_array(float * d_array_p, float value, int size)
{
	/*dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_fill_array<<<blocks_per_grid, threads_per_block>>>(d_array_p, value, size);*/
	cuda_safe_call(cudaMemset(d_array_p, value, size * sizeof(float)));
}

void add_matrices(float * d_input_p, float * d_out, float * d_bias_p, int size, int num)
{
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, num);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_add_matrices<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_bias_p, d_out, size);
}

/*
void multiply_matrices(float * d_input_p, float * d_out, float * d_partial_outputs, float * d_weights_p, int rows, int cols, int num)
{	
	unsigned int reduce_size = ceil_div(MATMUL_BLOCK_SIZEx2, cols);

	dim3 threads_per_block(MATMUL_BLOCK_SIZE, 1, INPUT_CHUNK_SIZE);
	dim3 blocks_per_grid(reduce_size, ceil_div(MAT_ROWS_PER_THREAD, rows), ceil_div(INPUT_CHUNK_SIZE, num));

	d_multiply_matrix_vector<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_weights_p, d_partial_outputs, cols, rows, cols, num);
	
	blocks_per_grid = dim3(num, ceil_div(MAT_ROWS_PER_THREAD, rows));

	if (reduce_size <= 2) {
		threads_per_block = dim3(1, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<1> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 4) {
		threads_per_block = dim3(2, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<2> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 8) {
		threads_per_block = dim3(4, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<4> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 16) {
		threads_per_block = dim3(8, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<8> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 32) {
		threads_per_block = dim3(16, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<16> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 64) {
		threads_per_block = dim3(32, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<32> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 128) {
		threads_per_block = dim3(64, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<64> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 256) {
		threads_per_block = dim3(128, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<128> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 512) {
		threads_per_block = dim3(256, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<256> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else if (reduce_size <= 1024) {
		threads_per_block = dim3(512, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<512> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, rows);
	}
	else {
		throw new exception("Matrix too large to multiply");
	}
}
*/

void multiply_matrices(float * d_A, float * d_B, float * d_partial_outputs, float * d_out, int A_rows, int A_cols, int B_rows, int B_cols)
{
	unsigned int partial_sum_count = ceil_div(M_TILE_SIZE_X, A_cols);

	//dim3 threads_per_block(M_TILE_SIZE_X, M_TILE_SIZE_X, 1);
	//dim3 blocks_per_grid(partial_sum_count, ceil_div(M_TILE_SIZE_X, A_rows), ceil_div(M_TILE_SIZE_X, B_cols));
	dim3 threads_per_block(M_TILE_SIZE_X, M_TILE_SIZE_Y, 1);
	dim3 blocks_per_grid(partial_sum_count, ceil_div(M_TILE_SIZE_Y, A_rows), ceil_div(M_TILE_SIZE_X, B_cols));

	if (partial_sum_count == 1) {
		d_multiply_matrices<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_out, A_rows, A_cols, B_rows, B_cols);
		return;
	}

	d_multiply_matrices<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_partial_outputs, A_rows, A_cols, B_rows, B_cols);

	cuda_safe_call(cudaDeviceSynchronize());

	//add partial sums for large matrices.

	if (partial_sum_count <= 2) {
		dim3 threads_per_block(1024, 1);
		dim3 blocks_per_grid(ceil_div(1024, A_rows * B_cols));
		d_compute_sums<2><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 4) {
		dim3 threads_per_block(512, 2);
		dim3 blocks_per_grid(ceil_div(512, A_rows * B_cols));
		d_compute_sums<4><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 8) {
		dim3 threads_per_block(256, 4);
		dim3 blocks_per_grid(ceil_div(256, A_rows * B_cols));
		d_compute_sums<8><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 16) {
		dim3 threads_per_block(128, 8);
		dim3 blocks_per_grid(ceil_div(128, A_rows * B_cols));
		d_compute_sums<16><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 32) {
		dim3 threads_per_block(64, 16);
		dim3 blocks_per_grid(ceil_div(64, A_rows * B_cols));
		d_compute_sums<32><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 64) {
		dim3 threads_per_block(32, 32);
		dim3 blocks_per_grid(ceil_div(32, A_rows * B_cols));
		d_compute_sums<64><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 128) {
		dim3 threads_per_block(16, 64);
		dim3 blocks_per_grid(ceil_div(16, A_rows * B_cols));
		d_compute_sums<128><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 256) {
		dim3 threads_per_block(8, 128);
		dim3 blocks_per_grid(ceil_div(8, A_rows * B_cols));
		d_compute_sums<256><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 512) {
		dim3 threads_per_block(4, 256);
		dim3 blocks_per_grid(ceil_div(4, A_rows * B_cols));
		d_compute_sums<512><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 1024) {
		dim3 threads_per_block(2, 512);
		dim3 blocks_per_grid(ceil_div(2, A_rows * B_cols));
		d_compute_sums<1024><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else if (partial_sum_count <= 2048) {
		dim3 threads_per_block(1, 1024);
		dim3 blocks_per_grid(ceil_div(1, A_rows * B_cols));
		d_compute_sums<2048><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, partial_sum_count, A_rows * B_cols);
	}
	else {
		throw new exception("Matrix too large to multiply");
	}

	/*unsigned int reduce_size = ceil_div(MATMUL_BLOCK_SIZEx2, A_cols);

	dim3 threads_per_block(MATMUL_BLOCK_SIZE, 1, INPUT_CHUNK_SIZE);
	dim3 blocks_per_grid(reduce_size, ceil_div(MAT_ROWS_PER_THREAD, A_rows), ceil_div(INPUT_CHUNK_SIZE, B_cols));

	d_multiply_matrices<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_partial_outputs, A_rows, A_cols, B_rows, B_cols, A_rows);

	blocks_per_grid = dim3(B_cols, ceil_div(MAT_ROWS_PER_THREAD, A_rows));

	if (reduce_size <= 2) {
		threads_per_block = dim3(1, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<1><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 4) {
		threads_per_block = dim3(2, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<2><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 8) {
		threads_per_block = dim3(4, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<4><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 16) {
		threads_per_block = dim3(8, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<8><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 32) {
		threads_per_block = dim3(16, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<16><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 64) {
		threads_per_block = dim3(32, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<32><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 128) {
		threads_per_block = dim3(64, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<64><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 256) {
		threads_per_block = dim3(128, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<128><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 512) {
		threads_per_block = dim3(256, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<256><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else if (reduce_size <= 1024) {
		threads_per_block = dim3(512, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<512><<<blocks_per_grid, threads_per_block>>>(d_partial_outputs, d_out, A_rows, reduce_size, A_rows);
	}
	else {
		throw new exception("Matrix too large to multiply");
	}*/
}

void multiply_staggered(float * d_input_p, float * d_out, float * d_partial_outputs, float * d_mul_mat, int rows, int cols, int stagger_size, int num)
{
	unsigned int reduce_size = ceil_div(MATMUL_BLOCK_SIZEx2, cols);

	dim3 threads_per_block(MATMUL_BLOCK_SIZE, 1, INPUT_CHUNK_SIZE);
	dim3 blocks_per_grid(reduce_size, ceil_div(MAT_ROWS_PER_THREAD, rows), ceil_div(INPUT_CHUNK_SIZE, num));

	d_multiply_matrix_vector << <blocks_per_grid, threads_per_block >> >(d_input_p, d_mul_mat, d_partial_outputs, cols, rows, stagger_size, num);

	blocks_per_grid = dim3(num, ceil_div(MAT_ROWS_PER_THREAD, rows));

	if (reduce_size <= 2) {
		threads_per_block = dim3(1, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<1> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 4) {
		threads_per_block = dim3(2, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<2> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 8) {
		threads_per_block = dim3(4, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<4> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 16) {
		threads_per_block = dim3(8, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<8> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 32) {
		threads_per_block = dim3(16, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<16> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 64) {
		threads_per_block = dim3(32, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<32> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 128) {
		threads_per_block = dim3(64, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<64> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 256) {
		threads_per_block = dim3(128, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<128> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 512) {
		threads_per_block = dim3(256, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<256> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else if (reduce_size <= 1024) {
		threads_per_block = dim3(512, MAT_ROWS_PER_THREAD);
		d_reduce_matvec_rows<512> << <blocks_per_grid, threads_per_block >> >(d_partial_outputs, d_out, rows, reduce_size, stagger_size);
	}
	else {
		throw new exception("Matrix too large to multiply");
	}
}

void apply_relu(float * d_input_p, float * d_output_p, int size, float alpha)
{
	//cuda_safe_call(cudaMemcpy(d_output_p, d_input_p, sizeof(float) * size, cudaMemcpyDeviceToDevice));
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_apply_relu<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size, alpha);
}

void apply_softmax(float * d_input_p, float * d_output_p, int input_size, int num, float beta)
{
	/*float * test = (float *)malloc(sizeof(float) * 10);
	cudaMemcpy(test, d_input_p, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++)
		printf("Softmax test[%d] = %f\n", i, test[i]);*/

	dim3 blocks_per_grid(1, num);
	if (input_size > SOFTMAX_MAX_CLASSES) {
		throw new exception("Too many classes in softmax function");
	}
	if (input_size <= 2) {
		dim3 threads_per_block(1, 1);
		d_apply_softmax<1><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 4) {
		dim3 threads_per_block(2, 1);
		d_apply_softmax<2><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 8) {
		dim3 threads_per_block(4, 1);
		d_apply_softmax<4><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 16) {
		dim3 threads_per_block(8, 1);
		d_apply_softmax<8><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 32) {
		dim3 threads_per_block(16, 1);
		d_apply_softmax<16><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 64) {
		dim3 threads_per_block(32, 1);
		d_apply_softmax<32><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 128) {
		dim3 threads_per_block(64, 1);
		d_apply_softmax<64><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 256) {
		dim3 threads_per_block(128, 1);
		d_apply_softmax<128><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 512) {
		dim3 threads_per_block(256, 1);
		d_apply_softmax<256><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 1024) {
		dim3 threads_per_block(512, 1);
		d_apply_softmax<512><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
	else if (input_size <= 2048) {
		dim3 threads_per_block(1024, 1);
		d_apply_softmax<1024><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, input_size, beta);
	}
}

void relu_derivative(float * d_input_p, float * d_output_p, int size, float alpha)
{
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	d_relu_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size, alpha);
}

void batch_norm(float * d_input_p, float * d_output_p, int size, int num)
{

}

void batch_norm_derivative(float * d_input_p, float * d_output_p, int size, int num)
{

}

/*void softmax_derivative(float * d_input_p, float * d_output_p, int size, float beta)
{
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	d_softmax_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size, beta);
}*/

void hadamard_product(float * d_a, float * d_b, float * d_output_p, int size)
{
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	d_hadamard_product<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_output_p, size);
}

void transpose(float * d_matrix_p, float * d_output_p, int rows, int cols)
{
	dim3 threads_per_block(cols, rows);
	dim3 blocks_per_grid(1, 1);
	if (cols * rows > 1024) {
		threads_per_block.x = 32;
		threads_per_block.y = 32;
		blocks_per_grid.x = ceil_div(32, cols);
		blocks_per_grid.y = ceil_div(32, rows);
	}
	d_transpose<<<blocks_per_grid, threads_per_block>>>(d_matrix_p, d_output_p, rows, cols);
}

/*void distributive_hadamard_transpose(float * d_input_p, float * d_matrix_p, float * d_output_p, int mat_rows, int mat_colse, int cols, int vector_size, int num)
{
}*/

void average_vector(float * d_matrix, float * d_output_p, int size, int num, int divisor)
{
	dim3 blocks_per_grid(1, size);

	if (num <= 2) {
		dim3 threads_per_block(1, 1);
		d_average_vector<1><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 4) {
		dim3 threads_per_block(2, 1);
		d_average_vector<2><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 8) {
		dim3 threads_per_block(4, 1);
		d_average_vector<4><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 16) {
		dim3 threads_per_block(8, 1);
		d_average_vector<8><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 32) {
		dim3 threads_per_block(16, 1);
		d_average_vector<16><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 64) {
		dim3 threads_per_block(32, 1);
		d_average_vector<32><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 128) {
		dim3 threads_per_block(64, 1);
		d_average_vector<64><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 256) {
		dim3 threads_per_block(128, 1);
		d_average_vector<128><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 512) {
		dim3 threads_per_block(256, 1);
		d_average_vector<256><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 1024) {
		dim3 threads_per_block(512, 1);
		d_average_vector<512><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else if (num <= 2048) {
		dim3 threads_per_block(1024, 1);
		d_average_vector<1024><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, size, num, divisor);
	}
	else {
		throw new exception("Batch size too large.");
	}
}

template <typename T_i, typename T_o>
extern void scalar_matrix_multiply(T_i * d_matrix, T_o * d_output_p, float scalar, int size)
{
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, 1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	d_scalar_matrix_mul<T_i, T_o><<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, scalar, size);
}

template void scalar_matrix_multiply<float, float>(float *, float *, float, int);
template void scalar_matrix_multiply<unsigned char, float>(unsigned char *, float *, float, int);

void average_value(float * d_input_p, float * average, int size)
{
	average_value(d_input_p, average, size, size);
}

void average_value(float * d_input_p, float * average, int size, float divisor)
{
	dim3 blocks_per_grid(1);
	
	if (size <= 2) {
		dim3 threads_per_block(1);
		d_average_value<1><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 4) {
		dim3 threads_per_block(2);
		d_average_value<2><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 8) {
		dim3 threads_per_block(4);
		d_average_value<4><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 16) {
		dim3 threads_per_block(8);
		d_average_value<8><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 32) {
		dim3 threads_per_block(16);
		d_average_value<16><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 64) {
		dim3 threads_per_block(32);
		d_average_value<32><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 128) {
		dim3 threads_per_block(64);
		d_average_value<64><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 256) {
		dim3 threads_per_block(128);
		d_average_value<128><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 512) {
		dim3 threads_per_block(256);
		d_average_value<256><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 1024) {
		dim3 threads_per_block(512);
		d_average_value<512><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else if (size <= 2048) {
		dim3 threads_per_block(1024);
		d_average_value<1024><<<blocks_per_grid, threads_per_block>>>(d_input_p, average, size, divisor);
	}
	else {
		throw new exception("Average matrix size too large");
	}
}

void subtract_partial_derivatives(float * d_matrix, float * d_derivatives, int size, float learning_rate)
{
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, 1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_sub_pds<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_derivatives, d_matrix, size, learning_rate);
}

void squared_error_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size)
{
	dim3 threads_per_block(BLOCK_SIZE);
	dim3 blocks_per_grid(ceil_div(BLOCK_SIZE * 2, size));

	d_squared_error<BLOCK_SIZE><<<blocks_per_grid, threads_per_block, BLOCK_SIZE * sizeof(float)>>>(d_input_p, d_target_p, d_output_p, size);
}

void squared_error_cost_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size)
{
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	d_squared_error_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size);
}

void softmax_cross_entropy_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num)
{
	dim3 threads_per_block(1, 1);
	dim3 blocks_per_grid(1, num);
	
	if (size <= 2) {
		threads_per_block.x = 1;
		d_softmax_cross_entropy<1><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 4) {
		threads_per_block.x = 2;
		d_softmax_cross_entropy<2><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 8) {
		threads_per_block.x = 4;
		d_softmax_cross_entropy<4><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 16) {
		threads_per_block.x = 8;
		d_softmax_cross_entropy<8><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 32) {
		threads_per_block.x = 16;
		d_softmax_cross_entropy<16><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 64) {
		threads_per_block.x = 32;
		d_softmax_cross_entropy<32><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 128) {
		threads_per_block.x = 64;
		d_softmax_cross_entropy<64><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 256) {
		threads_per_block.x = 128;
		d_softmax_cross_entropy<128><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 512) {
		threads_per_block.x = 256;
		d_softmax_cross_entropy<256><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 1024) {
		threads_per_block.x = 512;
		d_softmax_cross_entropy<512><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 2048) {
		threads_per_block.x = 1024;
		d_softmax_cross_entropy<1024><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}

	/*if (size <= 2) {
		dim3 threads_per_block(1, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 1024) {
			threads_per_block.y = 1024;
			blocks_per_grid.y = ceil_div(1024, num);
		}
		d_softmax_cross_entropy<1><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 4) {
		dim3 threads_per_block(2, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 512) {
			threads_per_block.y = 512;
			blocks_per_grid.y = ceil_div(512, num);
		}
		d_softmax_cross_entropy<2><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 8) {
		dim3 threads_per_block(4, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 256) {
			threads_per_block.y = 256;
			blocks_per_grid.y = ceil_div(256, num);
		}
		d_softmax_cross_entropy<4><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 16) {
		dim3 threads_per_block(8, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 128) {
			threads_per_block.y = 128;
			blocks_per_grid.y = ceil_div(128, num);
		}
		d_softmax_cross_entropy<8><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 32) {
		dim3 threads_per_block(16, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 64) {
			threads_per_block.y = 64;
			blocks_per_grid.y = ceil_div(64, num);
		}
		d_softmax_cross_entropy<16><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 64) {
		dim3 threads_per_block(32, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 32) {
			threads_per_block.y = 32;
			blocks_per_grid.y = ceil_div(32, num);
		}
		d_softmax_cross_entropy<32><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 128) {
		dim3 threads_per_block(64, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 16) {
			threads_per_block.y = 16;
			blocks_per_grid.y = ceil_div(16, num);
		}
		d_softmax_cross_entropy<64><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 256) {
		dim3 threads_per_block(128, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 8) {
			threads_per_block.y = 8;
			blocks_per_grid.y = ceil_div(8, num);
		}
		d_softmax_cross_entropy<128><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 512) {
		dim3 threads_per_block(256, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 4) {
			threads_per_block.y = 4;
			blocks_per_grid.y = ceil_div(4, num);
		}
		d_softmax_cross_entropy<256><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 1024) {
		dim3 threads_per_block(512, num);
		dim3 blocks_per_grid(1, 1);
		if (num > 2) {
			threads_per_block.y = 2;
			blocks_per_grid.y = ceil_div(2, num);
		}
		d_softmax_cross_entropy<512><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else if (size <= 2048) {
		dim3 threads_per_block(1024, 1);
		dim3 blocks_per_grid(1, num);
		d_softmax_cross_entropy<1024><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size, num);
	}
	else {
		throw new exception("Too many softmax classes");
	}*/
}

void softmax_cross_entropy_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num)
{
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, num);
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	d_softmax_cross_entropy_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size);
}

#define TEST_SIZE_X 64
#define TEST_SIZE_Y 16

__global__ void d_TEST_load_time(float * input0, float * input1) {
	__shared__ float s_mem0[TEST_SIZE_X * TEST_SIZE_Y];
	__shared__ float s_mem1[TEST_SIZE_X * TEST_SIZE_Y];

	unsigned int tid = threadIdx.x + threadIdx.y * TEST_SIZE_X;
	unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;

	unsigned int gid = bid * TEST_SIZE_X * TEST_SIZE_Y + tid;

	s_mem0[tid] = input0[gid];
	s_mem1[tid] = input1[gid];

	cuda_syncthreads();
}
