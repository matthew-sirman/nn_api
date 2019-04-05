#pragma once

#include <assert.h>

#include "tensor.h"
#include "shape.h"

//Base class for iterable data which should be divided into batches
//This base class can be inherited from to pass a dataset into a networks training
//function. So long as the pure virtual methods are overridden, the network model
//will handle accessing the batches for training.
class batch_iterator
{
public:
	//Default Constructor
	batch_iterator() {};

	//Destructor
	~batch_iterator() {};

	//Virtual method to create the next batch
	//This method can be used if the dataset requires (or is optimal) loading
	//the data and labels simultaneously. These should then be cached and
	//return with the getters.
	//If this is not necessary, this function does not need to be overloaded
	virtual void next_batch() {};

	//Virtual method to get the next batch of data
	//Must be overridden, and should return the next batch of data as a tensor
	virtual tensor * get_next_batch() = 0;

	//Virtual method to get the next batch of labels
	//Must be overridden, and should return the next batch of labels as a tensor
	virtual tensor * get_next_batch_labels() = 0;

	//Reset Iterator
	//Called after every epoch to reset the iterator to the start of the dataset
	virtual void reset_iterator() = 0;

	//Get Size
	//Returns the number of items in the dataset
	size_t get_size() { return n_items; }

	//Get Batch Size
	//Returns the size of each batch to be loaded
	size_t get_batch_size() { return batch_size; }

	//Initialise
	//Must be overridden, initialises the iterator by specifying the batch size
	virtual void initialise(size_t batch_size) = 0;

	//Close
	//Must be overridden, uninitialises the iterator by freeing memory
	virtual void close() = 0;

	//Classes
	//Returns the number of label classes the model has
	size_t classes() { return n_classes; }
protected:
	//flag to indicate if the model has been initialised (to avoid reinitialising or
	//free undeclared memory addresses)
	bool initialised = false;

	//variable to hold batch size
	size_t batch_size;

	//variable to hold number of items in the dataset (initialised to -1 to show it hasn't
	//been loaded yet)
	size_t n_items = -1;

	//variable to hold the number of avaiable label classes
	size_t n_classes;
};

