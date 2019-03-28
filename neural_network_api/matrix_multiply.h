#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

//#include <cuda.h>
//#include <cuda_runtime.h>

//#include "kernel.h"
#include "float_ops.h"
#include "d_matrix.h"

constexpr auto BLOCK_TILE_N = 128;
constexpr auto BLOCK_TILE_M = 32;
constexpr auto BLOCK_TILE_K = 16;
constexpr auto THREAD_TILE_N = 8;
constexpr auto THREAD_TILE_M = 8;
constexpr auto THREAD_TILE_K = 2;
constexpr auto THREAD_SIZE_N = BLOCK_TILE_N / THREAD_TILE_N;
constexpr auto THREAD_SIZE_M = BLOCK_TILE_M / THREAD_TILE_M;

template <typename m_T, int tile_n, int tile_m, int tile_k>
extern NN_LIB_API __device__ void d_thread_level_multiply(m_T r_accum[tile_n * tile_m], m_T r_A[tile_m * tile_k], m_T r_B[tile_n * tile_k]);

template <typename m_T, order A_o = order::ROW, order B_o = order::ROW, order C_o = order::ROW>
extern NN_LIB_API void matrix_multiply(d_matrix<m_T> &A, d_matrix<m_T> &B, d_matrix<m_T> &C);