#pragma once

#include "instruction_functions.h"

using namespace nnet;

namespace nnet {
	//Base Layer
	//A base struct for defining a network layer (a cluster of different
	//functions which make up a layer)
	struct base_layer {
		//Lock
		//Lock the trainable functions in this layer and flag the layer as
		//locked
		virtual void lock() { is_locked = true; }

		//Unlock
		//Unlock the trainable functions in this layer and flag the layer
		//as unlocked
		virtual void unlock() { is_locked = false; }

		//Locked
		//Return if this layer is locked or not
		bool locked() { return is_locked; }
	protected:
		//Is Locked
		//Flag to indicate that the entire layer is locked
		//Note: It is still possible that the inner functions could be
		//separately locked and unlocked, but the layer lock and unlocks
		//will override.
		bool is_locked = false;
	};

	//Dense Layer
	//A dense layer containing both an addition function and a matrix multiplication
	//function
	struct dense_layer : public base_layer {
		dense_layer(add_function* add, matmul_function* matmul) { this->add = add; this->matmul = matmul; }

		//Add
		//Pointer to the addition function for this dense layer
		add_function* add = nullptr;

		//Mat Mul
		//Pointer to the matrix multiplication function for this dense layer
		matmul_function* matmul = nullptr;

		//Lock
		//Lock the trainable functions in this layer and flag the layer as
		//locked
		void lock() override {
			base_layer::lock();
			add->lock();
			matmul->lock();
		}

		//Unlock
		//Unlock the trainable functions in this layer and flag the layer
		//as unlocked
		void unlock() override {
			base_layer::unlock();
			add->unlock();
			matmul->unlock();
		}
	};
}