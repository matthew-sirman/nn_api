#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include <assert.h>

#include "tensor.h"

class NN_LIB_API batch_iterator
{
public:
	batch_iterator() {};
	~batch_iterator() {};

	virtual void next_batch() {};
	virtual tensor * get_next_batch() = 0;
	virtual tensor * get_next_batch_labels() = 0;
	virtual void reset_iterator() = 0;
	size_t get_size() { return n_items; }
	size_t get_batch_size() { return batch_size; }
	virtual void initialise(size_t batch_size) = 0;
	virtual void close() = 0;

	size_t classes() { return n_classes; }
protected:
	bool initialised = false;
	size_t batch_size;
	size_t n_items = -1;
	size_t n_classes;
};

