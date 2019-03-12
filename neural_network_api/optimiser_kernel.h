#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include "kernel.h"

void adam_update_momentum(
	float * d_momentum, 
	float * d_derivatives, 
	float beta, 
	size_t size);
void adam_update_velocity(
	float * d_velocity, 
	float * d_derivatives, 
	float beta, 
	size_t size);
void adam_update_parameters(
	float * d_params, 
	float * d_momentum, 
	float * d_velocity, 
	float learning_rate, 
	float beta1, 
	float beta2, 
	float epsilon,
	int t_step, 
	size_t size);