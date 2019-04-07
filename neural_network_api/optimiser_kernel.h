#pragma once

#include "kernel.h"

//API FUNCTION
//Adam Update Momentum
//Updates the momentum vector from one training
//step to the next
void adam_update_momentum(
	float * d_momentum, 
	float * d_derivatives, 
	double beta, 
	size_t size);

//API FUNCTION
//Adam Update Velocity
//Updates the velocity vector from one training
//step to the next
void adam_update_velocity(
	float * d_velocity, 
	float * d_derivatives, 
	double beta,
	size_t size);

//API FUNCTION
//Adam Update Parameters
//Updates the training tensor from one training
//step to the next
void adam_update_parameters(
	float * d_params, 
	float * d_momentum, 
	float * d_velocity, 
	double learning_rate,
	double beta1,
	double beta2,
	double epsilon,
	int t_step, 
	size_t size);