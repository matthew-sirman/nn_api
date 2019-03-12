#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include "tensor.h"

class NN_LIB_API batch_iterator
{
public:
	batch_iterator() {};
	~batch_iterator() {};

	virtual tensor * get_next_batch() = 0;
	virtual tensor * get_next_batch_labels() = 0;
	virtual void reset_iterator() = 0;
	virtual size_t get_size() = 0;
	virtual size_t get_batch_size() = 0;
	virtual void initialise(size_t batch_size) = 0;
protected:
	bool initialised = false;
};

