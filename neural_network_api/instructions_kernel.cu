#include "instructions_kernel.h"

//kernel function for minimalistic copying with minimal overhead
__global__ void d_fast_copy(float *input, float *output, int size) {
	//get the thread id 
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//if in range, copy the input to the output
	if (id < size)
		output[id] = input[id];
}

//kernel function to scale and offset an array of floats
__global__ void d_random_scale_offset(float *arr, float scale, float offset, int size) {
	//gets the thread id
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//if in range, scale and offset
	if (id < size)
		arr[id] = arr[id] * scale + offset;
}

//kernel function to fill an array with a constant on the device
__global__ void d_fill_array(float * arr, float value, int size) {
	//gets the thread id
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//if in range, set the value
	if (id < size)
		arr[id] = value;
}

//kernel function to add a bias vector to each input vector
__global__ void d_add_matrices(float *input, float *biases, float *output, int size) {
	//get the thread id
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//get the element
	int nid = blockIdx.y;
	int off = nid * size;

	//if the id is in range set the output to the sum of the inputs
	if (id < size)
		output[id + off] = input[id + off] + biases[id];
}

//kernel function to apply ReLU activation to the input
__global__ void d_apply_relu(float *input, float *output, int size, float alpha) {
	//get the id
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//check if the id is in range
	if (id < size) {
		//set the output to input when input > 0
		//or alpha * input when input <= 0
		if (input[id] > 0)
			output[id] = input[id];
		else
			output[id] = alpha * input[id];
	}
}

//kernel function to apply Tanh activation to the input
__global__ void d_apply_tanh(float * input, float * output, int size) {
	//get the id
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//if in range set the output to the hyperbolic tangent of the input
	if (id < size) {
		output[id] = tanhf(input[id]);
	}
}

//kernel function to apply Sigmoid activation to the input
__global__ void d_apply_sigmoid(float * input, float * output, int size) {
	//get the id
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//if in range set the output to the sigmoid of the input
	if (id < size) {
		output[id] = 1 / (1 + expf(-input[id]));
	}
}

//kernel function to apply softmax over the input
template <unsigned int block_size>
__global__ void d_apply_softmax(float *input, float *output, int input_size, float beta) {
	//shared memory allocation to store the partial sums of each exponential
	__shared__ double s_sum[block_size * 2];

	//cache to remember the original of each of the same exponentials
	__shared__ double s_exp_vals[block_size * 2];

	//the elemental id and the input id
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int inid = blockIdx.y;
	
	//the load index
	unsigned int exp_index = inid * input_size + tid;

	//temporary variables to avoid repeat exponential calculations
	//we load 2 elements in per thread for optimal performance
	double tmp0 = 0;
	double tmp1 = 0;

	//load either 1 or 2 values into the temp variables depending on whether the indices are in range
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

	//set the sum and cache memory at the right places to the temporary variables
	s_sum[tid + block_size] = tmp0;
	s_sum[tid] = tmp1;
	s_exp_vals[tid + block_size] = tmp0;
	s_exp_vals[tid] = tmp1;

	//synchronise to complete loading 
	cuda_syncthreads();

	//parallel reduction to find the sum of the "s_sum" array across many threads using binary
	//tree strided reduction (each thread adds powers of 2 ahead of itself to the current value
	//so by the end the total is stored in memory location 0)
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

	//check if one or both of the current indices are in range and write the corresponding exponential
	//divided by the total to the output, so softmax(xi) = e^xi/Sum[e^x]
	if (tid + block_size < input_size) {
		output[exp_index + block_size] = s_exp_vals[tid + block_size] / s_sum[0];
		output[exp_index] = s_exp_vals[tid] / s_sum[0];
	}
	else if (tid < input_size) {
		output[exp_index] = s_exp_vals[tid] / s_sum[0];
	}
}

//kernel function to calculate the derivative of the ReLU activation function
__global__ void d_relu_derivative(float * input, float * output, int size, float alpha) {
	//get the current id
	unsigned int tid = threadIdx.x + BLOCK_SIZE * blockIdx.x;

	//check the id is in range
	if (tid < size) {
		//if the input > 0, the ReLU would output x.
		//dx/dx = 1 therefore the derivative is always 1.
		if (input[tid] > 0)
			output[tid] = 1;
		//if the input <= 0, the ReLU would output alpha * x.
		//d[alpha * x]/dx = alpha therefore the derivative is always alpha.
		else
			output[tid] = alpha;
	}
}

//kernel function to calculate the derivative of the Tanh activation function
__global__ void d_tanh_derivative(float * input, float * output, int size) {
	//get the id
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//check the id is in range
	if (id < size) {
		//d[tanh(x)]/dx = sech(x)^2
		//				= 1 - tanh(x)^2
		//so calculate tanh(x) and cache to prevent
		//calculating twice and return 1 - tanh(x)^2
		float tmp = tanhf(input[id]);
		output[id] = 1 - tmp * tmp;
	}
}

//kernel function to calculate the derivative of the Sigmoid activation function
__global__ void d_sigmoid_derivative(float * input, float * output, int size) {
	//get the id
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	//check the id is in range
	if (id < size) {
		//d[1/(1+e^-x)]/dx = e^x/(1+e^-x)^2
		//				   = (1+e^-x)/(1+e^-x)^2 - 1/(1+e^-x)^2
		//				   = 1/(1+e^-x) - (1/(1+e^-x))^2
		//sigmoid: s(x) = 1/(1+e^-x)
		//so, d[1/(1+e^-x)]/dx = s(x) - s(x)^2
		//					   = s(x) (1 - s(x))
		//so calculate s(x) and cache
		//then return s(x) (1 - s(x))
		float tmp = 1 / (1 + expf(-input[id]));
		output[id] = tmp * (1 - tmp);
	}
}

//NOT IMPLEMENTED
template <unsigned int block_size>
__global__ void d_batch_norm(float * input, float * output, int size, int num) {
	__shared__ float s_mean[block_size * 2];
}

//NOT IMPLEMENTED
__global__ void d_batch_norm_derivative(float * input, float * output, int size, int num) {

}

//kernel to calculate the Hadamard product between two vectors (elementwise multiplication)
__global__ void d_hadamard_product(float * a, float * b, float * output, int size) {
	//get the id
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	//check the id is in range
	if (tid < size) {
		//output the product of the inputs, all at the id
		output[tid] = a[tid] * b[tid];
	}
}

//kernel to transpose an input matrix 
__global__ void d_transpose(float * input, float * output, int rows, int cols) {
	//get the ids for the current column and row
	unsigned int colid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int rowid = threadIdx.y + blockIdx.y * blockDim.y;

	//check both ids are in range
	if (colid < cols && rowid < rows) {
		//write the input to the output with the rows and columns flipped
		//such that each element is written to the transposed output
		output[colid * rows + rowid] = input[rowid * cols + colid];
	}
}

//kernel to calculate the average value of an input matrix where the output vector
//is the average of each row
template <unsigned int block_size>
__global__ void d_average_vector(float * input, float * output, int input_size, int num, int divisor) {
	//get the thread id and input id
	unsigned int tid = threadIdx.x;
	unsigned int g_inid = blockIdx.y;

	//allocate shared memory for the sums
	__shared__ float s_sums[block_size * 2];

	//load two elements into the shared memory provided they are in range
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

	//synchronise to complete loading before continuing
	cuda_syncthreads();

	//parallel reduction to find the sum of the "s_sums" array across many threads using binary
	//tree strided reduction (each thread adds powers of 2 ahead of itself to the current value
	//so by the end the total is stored in memory location 0)
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

	//if this is the first thread write the sum divided by the divisor
	//only writes out once per row, so only one thread writes, hence
	//the check
	if (tid == 0) {
		output[g_inid] = s_sums[0] / divisor;
	}
}

//kernel to multiply a float matrix by a scalar
__global__ void d_scalar_matrix_mul_f(float * input, float * output, float scalar, int size) {
	//get the id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//check if the id is in range
	if (tid < size) {
		//output the input at the id multiplied by the scalar
		output[tid] = input[tid] * scalar;
	}
}

//kernel to multiply a byte matrix by a scalar
__global__ void d_scalar_matrix_mul_b(byte * input, float * output, float scalar, int size) {
	//get the id
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//check if the id is in range
	if (tid < size) {
		//output the input at the id multiplied by the scalar
		output[tid] = input[tid] * scalar;
	}
}

//kernel to calculate the average value of an array
template <unsigned int block_size>
__global__ void d_average_value(float * input, float * output, int size, float divisor) {
	//allocate shared memory for each element
	__shared__ float s_sum[block_size * 2];

	//get 2 ids to load multiple data at once for performance
	int id0 = threadIdx.x;
	int id1 = id0 + block_size;

	//load 1 or 2 elements depending on whether they are within range
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

	//synchronise to complete loading
	cuda_syncthreads();

	//parallel reduction to find the sum of the "s_sum" array across many threads using binary
	//tree strided reduction (each thread adds powers of 2 ahead of itself to the current value
	//so by the end the total is stored in memory location 0)
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

	//write the sum divided by the divisor in one thread only
	if (id0 == 0)
		output[0] = (float)s_sum[0] / divisor;
}

//kernel to calculate the squared error cost between two distributions
template <unsigned int block_size>
__global__ void d_squared_error(float *input, float *target, float *output, int size) {
	//allocate an array to be summed
	__shared__ float s_cost[block_size];

	//get the thread id and the load id
	int tid = threadIdx.x;
	int i = 2 * block_size * blockIdx.x + tid;
	
	//load two target differences into registers
	float diff[2];
	diff[0] = input[i] - target[i];
	diff[1] = input[i + block_size] - target[i + block_size];
	//add the squares of both differences to the cost array
	s_cost[tid] += diff[0] * diff[0] + diff[1] * diff[1];

	//synchronise to complete loading
	cuda_syncthreads();

	//parallel reduction to find the sum of the "s_cost" array across many threads using binary
	//tree strided reduction (each thread adds powers of 2 ahead of itself to the current value
	//so by the end the total is stored in memory location 0)
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

	//unroll the last warp as thread synchronisation is guaranteed
	if (tid < 32) {
		warp_reduce<block_size>(s_cost, tid);
	}

	//if this is the last thread write to the output
	if (tid == 0)
		output[0] += s_cost[0];
}

//kernel to calculate the derivative of the squared error
__global__ void d_squared_error_derivative(float * input, float * target, float * output, int size) {
	//get the id
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	//check the id is in range
	if (tid < size)
		//d[(x-y)^2]/dx = 2(x-y)
		//so return 2 times the difference
		output[tid] = 2 * (input[tid] - target[tid]);
}

//kernel to calculate the cross entropy between two distributions
template <unsigned int block_size>
__global__ void d_softmax_cross_entropy(float *input, float *target, float *output, int size, int num) {
	//allocate shared memory for the exponentials and elements
	__shared__ double s_exps[block_size * 2];
	__shared__ float s_cost[block_size * 2];

	//get 2 ids for multiple loads per thread
	unsigned int x_id0 = threadIdx.x;
	unsigned int x_id1 = x_id0 + block_size;

	//global memory ids
	unsigned int in_id = blockIdx.y;
	unsigned int in_index0 = in_id * size + x_id0;
	unsigned int in_index1 = in_id * size + x_id1;

	//check which ids are in range
	//cache the input loads in registers to avoid multiple loads
	//of the same address and write the exponential and value times
	//the target to the s_exps and s_cost arrays
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

	//synchronise to complete loading
	cuda_syncthreads();

	//parallel reduction to find the sum of the arrays across many threads using binary
	//tree strided reduction (each thread adds powers of 2 ahead of itself to the current value
	//so by the end the total is stored in memory location 0)
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

	//if the index is within range write the natural log
	//of the sum of exponentials - the sum of the targets 
	//times inputs to global memory. After lots of simplifications,
	//the cost comes to just this.
	if (x_id0 == 0 && in_id < num) {
		atomic_add(output, logf(s_exps[0]) - s_cost[0]);
	}
}

//kernel to calculate the derivative of the cross entropy between 2 distributions
__global__ void d_softmax_cross_entropy_derivative(float *input, float *target, float *output, int size) {
	//get the id
	unsigned int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int id = x_id + blockIdx.y * size;

	//check if the id is in range
	if (x_id < size) {
		//after lots of simplifications, the derivative of the softmax cross entropy
		//comes to just p - y (where p is softmax(x), and x and y are the predictions and targets)
		output[id] = input[id] - target[id];
	}
}

//kernel to calculate the argmax for each input in the input vector
__global__ void d_argmax(float * input, float * max_vals, int * output, int input_size) {
	//declare shared memory to hold each value
	__shared__ float s_values[BLOCK_SIZE * 2];
	__shared__ int s_indices[BLOCK_SIZE * 2];

	//get the id of this thread
	int tid = threadIdx.x;

	//get the offset for this input block
	int input_block = blockIdx.x * BLOCK_SIZE;

	//get the current element
	int element_id = blockIdx.y;

	//check which ids are in range and load corresponding
	//values from global memory
	if (tid + input_block +BLOCK_SIZE < input_size) {
		s_values[tid + BLOCK_SIZE] = input[input_size * element_id + input_block + tid + BLOCK_SIZE];
		s_values[tid] = input[input_size * element_id + input_block + tid];
	}
	else if (tid + input_block < input_size) {
		s_values[tid + BLOCK_SIZE] = FLOAT_MIN;
		s_values[tid] = input[input_size * element_id + input_block + tid];
	}
	else {
		s_values[tid + BLOCK_SIZE] = FLOAT_MIN;
		s_values[tid] = FLOAT_MIN;
	}

	s_indices[tid + BLOCK_SIZE] = input_block + tid + BLOCK_SIZE;
	s_indices[tid] = input_block + tid;

	//synchronise threads to complete loading
	cuda_syncthreads();

	//use parallel reduction to reduce the array, but instead of summing, compare the values and save the
	//maximum. This will result in the highest value being stored in the first index. A simultaneos array of
	//the corresponding indices is reduces alongside, so at each stage the indices array and values array will
	//correspond to each other. This means the argmax will be the first index of the indices array
	if (BLOCK_SIZE >= 1024) {
		if (tid < 1024) {
			if (s_values[tid + 1024] > s_values[tid]) {
				s_values[tid] = s_values[tid + 1024];
				s_indices[tid] = s_indices[tid + 1024];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 512) {
		if (tid < 512) {
			if (s_values[tid + 512] > s_values[tid]) {
				s_values[tid] = s_values[tid + 512];
				s_indices[tid] = s_indices[tid + 512];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 256) {
		if (tid < 256) {
			if (s_values[tid + 256] > s_values[tid]) {
				s_values[tid] = s_values[tid + 256];
				s_indices[tid] = s_indices[tid + 256];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 128) {
		if (tid < 128) {
			if (s_values[tid + 128] > s_values[tid]) {
				s_values[tid] = s_values[tid + 128];
				s_indices[tid] = s_indices[tid + 128];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 64) {
		if (tid < 64) {
			if (s_values[tid + 64] > s_values[tid]) {
				s_values[tid] = s_values[tid + 64];
				s_indices[tid] = s_indices[tid + 64];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 32) {
		if (tid < 32) {
			if (s_values[tid + 32] > s_values[tid]) {
				s_values[tid] = s_values[tid + 32];
				s_indices[tid] = s_indices[tid + 32];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 16) {
		if (tid < 16) {
			if (s_values[tid + 16] > s_values[tid]) {
				s_values[tid] = s_values[tid + 16];
				s_indices[tid] = s_indices[tid + 16];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 8) {
		if (tid < 8) {
			if (s_values[tid + 8] > s_values[tid]) {
				s_values[tid] = s_values[tid + 8];
				s_indices[tid] = s_indices[tid + 8];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 4) {
		if (tid < 4) {
			if (s_values[tid + 4] > s_values[tid]) {
				s_values[tid] = s_values[tid + 4];
				s_indices[tid] = s_indices[tid + 4];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 2) {
		if (tid < 2) {
			if (s_values[tid + 2] > s_values[tid]) {
				s_values[tid] = s_values[tid + 2];
				s_indices[tid] = s_indices[tid + 2];
			}
		}
		cuda_syncthreads();
	}
	if (BLOCK_SIZE >= 1) {
		if (tid < 1) {
			if (s_values[tid + 1] > s_values[tid]) {
				s_values[tid] = s_values[tid + 1];
				s_indices[tid] = s_indices[tid + 1];
			}
		}
		cuda_syncthreads();
	}

	//if this is the last thread...
	if (tid == 0) {
		//if the new max is greater than the old max, write the new max value
		//and its index out
		if (max_vals[element_id] < s_values[0]) {
			max_vals[element_id] = s_values[0];
			output[element_id] = s_indices[0];
		}
	}
}

//kernel to calculate the number of equal elements between two arrays of integers
__global__ void d_comp_eq(int * a, int * b, unsigned int * res, size_t size) {
	//get the id
	int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;

	//check the id is in range
	if (tid < size) {
		//if the elements are equal increment the counter
		if (a[tid] == b[tid]) {
			atomic_add(res, 1);
		}
	}
}

//kernel to calculate the number of equal elements between two arrays of integers
__global__ void d_comp_eq(int * a, float * b, unsigned int * res, size_t size) {
	//get the id
	int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;

	//check the id is in range
	if (tid < size) {
		//if the elements are equal increment the counter
		if (a[tid] == b[tid]) {
			atomic_add(res, 1);
		}
	}
}

template <unsigned int block_size>
__device__ void warp_reduce(volatile float *s_pr_array, int tid) {
	//reduce last warp with an offset without synchronising as #
	//the last warp operates in sync
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
	//reduce last warp without synchronising as the last warp operates
	//in sync
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
	//reduce last warp without synchronising as the last warp operates
	//in sync
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

void allocate_device_float_pointer(float ** d_pointer, size_t size)
{
	cuda_safe_call(cudaMallocManaged(d_pointer, sizeof(float) * size));
}

void deallocate_device_float_pointer(float * d_pointer)
{
	if (d_pointer != nullptr)
		cuda_safe_call(cudaFree(d_pointer));
}

void load_data_into_device(float * input_data, float * d_data_p, size_t size)
{
	cuda_safe_call(cudaMemcpy(d_data_p, input_data, sizeof(float) * size, cudaMemcpyHostToDevice));
}

void retrieve_output_data(float * output_data, float * d_data_p, size_t size)
{
	cuda_safe_call(cudaMemcpy(output_data, d_data_p, sizeof(float) * size, cudaMemcpyDeviceToHost));
}

void copy_into_device_array(float * input_data, float * d_data_p, size_t size, size_t offset)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	//launch the fast copy kernel
	d_fast_copy<<<blocks_per_grid, threads_per_block>>>(input_data, &d_data_p[offset], size);
}

void get_prng(curandGenerator_t * prng, int seed)
{
	//create a device PRNG
	curandCreateGenerator(prng, CURAND_RNG_PSEUDO_XORWOW);

	//setup the PRNG with the supplied seed
	curandSetPseudoRandomGeneratorSeed(*prng, seed);
}

void random_host_array(curandGenerator_t prng, float * array_p, float scale, float offset, size_t size)
{
	//create a placeholder device array and allocate it memory
	float *d_array_p;
	cuda_safe_call(cudaMallocManaged(&d_array_p, sizeof(float) * size));

	//generate a uniform distribution within the array
	curandGenerateUniform(prng, d_array_p, size);

	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}

	//launch the kernel to scale and offset the values in the array (as a standard uniform is between 0 and 1)
	d_random_scale_offset<<<blocks_per_grid, threads_per_block>>>(d_array_p, scale, offset, size);

	//copy the result back to the host
	cuda_safe_call(cudaMemcpy(array_p, d_array_p, sizeof(float) * size, cudaMemcpyDeviceToHost));

	//free the placeholder
	cuda_safe_call(cudaFree(d_array_p));
}

void random_normal_array(curandGenerator_t prng, float * array_p, float mean, float stddev, size_t size)
{
	//create a placeholder device array and allocate it memory
	float * d_array_p;
	cuda_safe_call(cudaMallocManaged(&d_array_p, sizeof(float) * size));

	//sample from a normal distribution to fill array
	curandGenerateNormal(prng, d_array_p, size, mean, stddev);

	//copy the result back to the host
	cuda_safe_call(cudaMemcpy(array_p, d_array_p, sizeof(float) * size, cudaMemcpyDeviceToHost));

	//free the placeholder
	cuda_safe_call(cudaFree(d_array_p));
}

void fill_device_array(float * d_array_p, float value, size_t size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	
	//launch the kernel to fill the array with the specified value
	d_fill_array<<<blocks_per_grid, threads_per_block>>>(d_array_p, value, size);
}

void add_matrices(float * d_input_p, float * d_out, float * d_bias_p, int size, int num)
{
	//setup block and grid sizes
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, num);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}

	//launch the add matrices kernel
	d_add_matrices<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_bias_p, d_out, size);
}

void apply_relu(float * d_input_p, float * d_output_p, int size, float alpha)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	//launch the ReLU kernel
	d_apply_relu<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size, alpha);
}

void apply_tanh(float * d_input_p, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	//launch the Tanh kernel
	d_apply_tanh<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size);
}

void apply_sigmoid(float * d_input_p, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	//launch the Sigmoid kernel
	d_apply_sigmoid<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size);
}

void apply_softmax(float * d_input_p, float * d_output_p, int input_size, int num, float beta)
{
	//setup block and grid sizes
	//divide the input size into differing power of 2 categories
	//each line will evaluate the template, so all the if checks in the kernel 
	//can be predetermined by the compiler, and the checks only need to be
	//made once on the host before calling
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
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the ReLU derivative kernel
	d_relu_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size, alpha);
}

void tanh_derivative(float * d_input_p, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the Tanh derivative kernel
	d_tanh_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size);
}

void sigmoid_derivative(float * d_input_p, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the Sigmoid derivative kernel
	d_sigmoid_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_output_p, size);
}

//NOT IMPLEMENTED
void batch_norm(float * d_input_p, float * d_output_p, int size, int num)
{

}

//NOT IMPLEMENTED
void batch_norm_derivative(float * d_input_p, float * d_output_p, int size, int num)
{

}

void hadamard_product(float * d_a, float * d_b, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the Hadamard product kernel
	d_hadamard_product<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_output_p, size);
}

void transpose(float * d_matrix_p, float * d_output_p, int rows, int cols)
{
	//setup block and grid sizes
	dim3 threads_per_block(cols, rows);
	dim3 blocks_per_grid(1, 1);
	//handle thread overflow
	if (cols * rows > 1024) {
		threads_per_block.x = 32;
		threads_per_block.y = 32;
		blocks_per_grid.x = ceil_div(32, cols);
		blocks_per_grid.y = ceil_div(32, rows);
	}
	//launch the transpose kernel
	d_transpose<<<blocks_per_grid, threads_per_block>>>(d_matrix_p, d_output_p, rows, cols);
}

void average_vector(float * d_matrix, float * d_output_p, int size, int num, int divisor)
{
	//setup block and grid sizes
	//divide the input size into differing power of 2 categories
	//each line will evaluate the template, so all the if checks in the kernel 
	//can be predetermined by the compiler, and the checks only need to be
	//made once on the host before calling
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

void scalar_matrix_multiply_f(float * d_matrix, float * d_output_p, float scalar, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, 1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the scalar matrix multiplication kernel
	d_scalar_matrix_mul_f<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, scalar, size);
}

void scalar_matrix_multiply_b(byte * d_matrix, float * d_output_p, float scalar, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, 1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the scalar matrix multiplication kernel
	d_scalar_matrix_mul_b<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_output_p, scalar, size);
}

void average_value(float * d_input_p, float * average, int size)
{
	//wrap the function assuming pure average is desired
	average_value(d_input_p, average, size, size);
}

void average_value(float * d_input_p, float * average, int size, float divisor)
{
	//setup block and grid sizes
	//divide the input size into differing power of 2 categories
	//each line will evaluate the template, so all the if checks in the kernel 
	//can be predetermined by the compiler, and the checks only need to be
	//made once on the host before calling
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

void squared_error_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(BLOCK_SIZE);
	dim3 blocks_per_grid(ceil_div(BLOCK_SIZE * 2, size));

	//launch the squared error kernel
	d_squared_error<BLOCK_SIZE><<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size);
}

void squared_error_cost_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil((float)size / BLOCK_SIZE);
	}
	//launch the squared error derivative kernel
	d_squared_error_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size);
}

void softmax_cross_entropy_cost(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num)
{
	//setup block and grid sizes
	//divide the input size into differing power of 2 categories
	//each line will evaluate the template, so all the if checks in the kernel 
	//can be predetermined by the compiler, and the checks only need to be
	//made once on the host before calling
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
}

void softmax_cross_entropy_derivative(float * d_input_p, float * d_target_p, float * d_output_p, int size, int num)
{
	//setup block and grid sizes
	dim3 threads_per_block(size, 1);
	dim3 blocks_per_grid(1, num);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the softmax cross entropy derivative kernel
	d_softmax_cross_entropy_derivative<<<blocks_per_grid, threads_per_block>>>(d_input_p, d_target_p, d_output_p, size);
}

void argmax(float * d_input, int * d_output, size_t input_size, size_t num)
{
	float * d_max_vals;

	cuda_safe_call(cudaMallocManaged(&d_max_vals, sizeof(float) * num));

	argmax(d_input, d_max_vals, d_output, input_size, num);

	cuda_safe_call(cudaFree(d_max_vals));
}

void argmax(float * d_input, float * d_max_vals, int * d_output, size_t input_size, size_t num)
{
	//fill the array with the minimum value a float can hold to initialise
	fill_device_array(d_max_vals, FLOAT_MIN, num);

	//setup block and grid sizes
	dim3 threads_per_block(BLOCK_SIZE, 1);
	dim3 blocks_per_grid(ceil_div(BLOCK_SIZE * 2, input_size), num);
	//launch the argmax kernel
	d_argmax<<<blocks_per_grid, threads_per_block>>>(d_input, d_max_vals, d_output, input_size);
}

void comp_eq(int * a, int * b, unsigned int * res, size_t size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the comparison equal kernel
	d_comp_eq<<<blocks_per_grid, threads_per_block>>>(a, b, res, size);
}

void comp_eq(int * a, float * b, unsigned int * res, size_t size)
{
	//setup block and grid sizes
	dim3 threads_per_block(size);
	dim3 blocks_per_grid(1);
	//handle thread overflow
	if (size > BLOCK_SIZE) {
		threads_per_block.x = BLOCK_SIZE;
		blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
	}
	//launch the comparison equal kernel
	d_comp_eq<<<blocks_per_grid, threads_per_block>>>(a, b, res, size);
}