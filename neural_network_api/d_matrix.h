#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernel.h"

#include "util.h"

enum NN_LIB_API order {
	COL,
	ROW
};

template <typename T>
struct NN_LIB_API __align__(8) d_matrix {
	int n_size;
	int m_size;

	T * d_data;

	inline d_matrix<T>() {};

	inline d_matrix<T>(int n_size, int m_size, T * d_data) {
		this->n_size = n_size;
		this->m_size = m_size;
		this->d_data = d_data;
	}

	__host__ __device__ T get(int n, int m) const {
		return d_data[m * n_size + n];
		/*switch (o_T) {
		case mat_order::ROW_MAJOR:
			return d_data[m * n_size + n];
		case mat_order::COLUMN_MAJOR:
			return d_data[n * m_size + m];
		default:
			return T();
		}*/
	}

	/*template <int w, int h, int b_w, int b_h>
	__host__ __device__ void stream_copy(volatile T * buff, block<w, h> buff_block, block<w, h> mat_block) {
		switch (o_T) {
		case mat_order::ROW_MAJOR:
			float4* f4_buff = reinterpret_cast<float4*>(buff);
			float4* df_mat;

			#pragma unroll
			for (int load_m = 0; load_m < h; load_m++) {
				df_mat = reinterpret_cast<float4*>(&d_data[((mat_block.y + load_m) * n_size) % 4 + (mat_block.x / 4)]);
				#pragma unroll
				for (int load_n = 0; load_n < w / 4; load_n++) {
					f4_buff[((buff_block.y + load_m) * w + buff_block.x) / 4 + load_n] = df_mat[load_n];
				}
				#pragma unroll
				for (int rem_load = 0; rem_load < w % 4; rem_load++) {

				}
			}
			break;
		case mat_order::COLUMN_MAJOR:
			float4* f4_buff = reinterpret_cast<float4*>(buff);
			float4* df_mat;

			#pragma unroll
			for (int load_m = 0; load_m < h; load_m++) {
				df_mat = reinterpret_cast<float4*>(&d_data[((mat_block.y + load_m) * n_size) % 4 + (mat_block.x / 4)]);
				#pragma unroll
				for (int load_n = 0; load_n < w / 4; load_n++) {
					f4_buff[((buff_block.y + load_m) * w + buff_block.x) / 4 + load_n] = df_mat[load_n];
				}
			}
			break;
		}
	}*/

	__host__ __device__ void set(int n, int m, T value) const {
		d_data[m * n_size + n] = value;
		/*switch (o_T) {
		case mat_order::ROW_MAJOR:
			d_data[m * n_size + n] = value;
			break;
		case mat_order::COLUMN_MAJOR:
			d_data[n * m_size + m] = value;
			break;
		default:
			break;
		}*/
	}
	__host__ __device__ void incr(int n, int m, T value) const {
		d_data[m * n_size + n] += value;
		/*switch (o_T) {
		case mat_order::ROW_MAJOR:
			d_data[m * n_size + n] += value;
			break;
		case mat_order::COLUMN_MAJOR:
			d_data[n * m_size + m] += value;
			break;
		default:
			break;
		}*/
	}
};

void cpy_f2_2d(float2 * dst, float * src, int n, int m);
void cpy_f4_2d(float4 * dst, float * src, int n, int m);

template <typename m_T>
extern NN_LIB_API inline void create_device_matrix(d_matrix<m_T> &matrix, int n_size, int m_size, m_T * d_data) {
	matrix.n_size = n_size;
	matrix.m_size = m_size;
	matrix.d_data = d_data;
}

extern NN_LIB_API inline void create_f1_matrix(d_matrix<float1> * matrix, int n_size, int m_size, float * data) {
	matrix = new d_matrix<float1>();

	matrix->n_size = n_size;
	matrix->m_size = m_size;

	cuda_safe_call(cudaMallocManaged(&matrix->d_data, sizeof(float1) * n_size * m_size));

	matrix->d_data = reinterpret_cast<float1*>(data);
}

extern NN_LIB_API inline void create_f2_matrix(d_matrix<float2> * matrix, int n_size, int m_size, float * data) {
	matrix = new d_matrix<float2>();

	matrix->n_size = ceil_div(2, n_size);
	matrix->m_size = 2 * ceil_div(2, m_size);

	cuda_safe_call(cudaMallocManaged(&matrix->d_data, sizeof(float2) * matrix->n_size * matrix->m_size));
	cpy_f2_2d(matrix->d_data, data, n_size, m_size);
}

extern NN_LIB_API inline void create_f4_matrix(d_matrix<float4> * matrix, int n_size, int m_size, float * data) {
	matrix = new d_matrix<float4>();

	matrix->n_size = ceil_div(4, n_size);
	matrix->m_size = 4 * ceil_div(4, m_size);

	cuda_safe_call(cudaMallocManaged(&matrix->d_data, sizeof(float4) * matrix->n_size * matrix->m_size));
	cpy_f4_2d(matrix->d_data, data, n_size, m_size);
}