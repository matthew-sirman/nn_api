#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernel.h"

#include "util.h"

//the order in which the matrix is indexed (column major or row major)
enum order {
	COL,
	ROW
};

//Device Matrix
//Declare a matrix of generic type T
//Matrix of size N*M on device memory for
//linear algebra operations
template <typename T>
struct __align__(8) d_matrix {
	//N size (cols) of the matrix
	//M size (rows) of the matrix
	int n_size;
	int m_size;

	//array data
	T * d_data;

	//Default constructor 
	inline d_matrix<T>() {};

	//Constructor specifying N size M size and an array of data on
	//the device
	inline d_matrix<T>(int n_size, int m_size, T * d_data) {
		this->n_size = n_size;
		this->m_size = m_size;
		this->d_data = d_data;
	}

	//Get
	//Get the element at a given index
	__host__ __device__ T get(int n, int m) const {
		return d_data[m * n_size + n];
	}

	//Set
	//Set the element at a given index
	__host__ __device__ void set(int n, int m, T value) const {
		d_data[m * n_size + n] = value;
	}

	//Increment
	//Increment a value at a given index by a value specified
	__host__ __device__ void incr(int n, int m, T value) const {
		d_data[m * n_size + n] += value;
	}
};

//Create Device Matrix
//Template function to create a device matrix
template <typename m_T>
inline void create_device_matrix(d_matrix<m_T> &matrix, int n_size, int m_size, m_T * d_data) {
	//setup parameters of the matrix
	matrix.n_size = n_size;
	matrix.m_size = m_size;
	matrix.d_data = d_data;
}

//Create F1 Matrix
//Create a matrix of floats from a host array
inline void create_f1_matrix(d_matrix<float1> * matrix, int n_size, int m_size, float * data) {
	matrix = new d_matrix<float1>();

	matrix->n_size = n_size;
	matrix->m_size = m_size;

	cuda_safe_call(cudaMallocManaged(&matrix->d_data, sizeof(float1) * n_size * m_size));

	matrix->d_data = reinterpret_cast<float1*>(data);
}