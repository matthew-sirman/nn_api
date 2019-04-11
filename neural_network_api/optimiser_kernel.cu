#include "optimiser_kernel.h"

namespace nnet {
	namespace nnet_internal {
		//kernel function to update the momentum
		__global__ void d_adam_update_momentum(
			float* momentum,
			float* derivatives,
			double beta,
			size_t size)
		{
			//get the id
			int tid = threadIdx.x + blockDim.x * blockIdx.x;

			//check the id is in range
			if (tid < size) {
				//update this specific momentum variable
				momentum[tid] = beta * momentum[tid] + (1 - beta) * derivatives[tid];
			}
		}

		//kernel function to update the velocity
		__global__ void d_adam_update_velocity(
			float* velocity,
			float* derivatives,
			double beta,
			size_t size)
		{
			//get the id
			int tid = threadIdx.x + blockDim.x * blockIdx.x;

			//check the id is in range
			if (tid < size) {
				//cache the input to reduce loads
				float tmp = derivatives[tid];

				//update this specific velocity variable
				velocity[tid] = beta * velocity[tid] + (1 - beta) * tmp * tmp;
			}
		}

		//kernel function to update the velocity
		__global__ void d_adam_update_parameters(
			float* params,
			float* momentum,
			float* velocity,
			double eta,
			double epsilon,
			size_t size)
		{
			//get the id
			int tid = threadIdx.x + blockDim.x * blockIdx.x;

			//check the id is in range
			if (tid < size) {
				//update this parameter velocity variable
				params[tid] = params[tid] - eta * (momentum[tid] / (sqrt(velocity[tid]) + epsilon));
			}
		}

		void adam_update_momentum(
			float* d_momentum,
			float* d_derivatives,
			double beta,
			size_t size)
		{
			//setup block and grid sizes
			dim3 threads_per_block(size);
			dim3 blocks_per_grid(1);
			//handle thread overflow
			if (size > BLOCK_SIZE) {
				threads_per_block.x = BLOCK_SIZE;
				blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
			}

			//launch the update momentum kernel
			d_adam_update_momentum << <blocks_per_grid, threads_per_block >> > (
				d_momentum,
				d_derivatives,
				beta,
				size
				);
		}

		void adam_update_velocity(
			float* d_velocity,
			float* d_derivatives,
			double beta,
			size_t size)
		{
			//setup block and grid sizes
			dim3 threads_per_block(size);
			dim3 blocks_per_grid(1);
			//handle thread overflow
			if (size > BLOCK_SIZE) {
				threads_per_block.x = BLOCK_SIZE;
				blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
			}

			//launch the update velocity kernel
			d_adam_update_velocity << <blocks_per_grid, threads_per_block >> > (
				d_velocity,
				d_derivatives,
				beta,
				size
				);
		}

		void adam_update_parameters(
			float* d_params,
			float* d_momentum,
			float* d_velocity,
			double learning_rate,
			double beta1,
			double beta2,
			double epsilon,
			int t_step,
			size_t size)
		{
			//setup block and grid sizes
			dim3 threads_per_block(size);
			dim3 blocks_per_grid(1);
			//handle thread overflow
			if (size > BLOCK_SIZE) {
				threads_per_block.x = BLOCK_SIZE;
				blocks_per_grid.x = ceil_div(BLOCK_SIZE, size);
			}

			//calculate the adapted learning rate
			double eta = learning_rate * (sqrtf(1 - pow(beta2, t_step + 1)) / (1 - pow(beta1, t_step + 1)));

			//launch the update parameters kernel
			d_adam_update_parameters << <blocks_per_grid, threads_per_block >> > (
				d_params,
				d_momentum,
				d_velocity,
				eta,
				epsilon,
				size
				);
		}
	}
}