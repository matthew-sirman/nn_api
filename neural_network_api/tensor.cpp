#include "stdafx.h"

#include "tensor.h"

namespace nnet {
	tensor::tensor(size_t size)
	{
		//check the size isn't 0
		if (size == 0)
			throw exception("Cannot have a tensor with size 0");

		//set the shape to be 1D and have the right size
		this->shape = { size };

		//allocate enough host memory for this tensor
		this->data = (float*)malloc(sizeof(float) * size);
	}

	tensor::tensor(size_t size, float* data)
	{
		//check the size isn't 0
		if (size == 0)
			throw exception("Cannot have a tensor with size 0");

		//set the shape to be 1D and have the right size
		this->shape = { size };

		//set the local data pointer to be the input data pointer
		this->data = data;
	}

	tensor::tensor(vector<size_t> shape)
	{
		//check the number of dimensions
		if (shape.size() == 0)
			throw exception("Shape must have at least 1 dimension");

		//set the shape to be the input shape
		this->shape = shape;

		//check the size isn't 0
		if (this->get_size() == 0)
			throw exception("Cannot have a tensor with size 0");

		//allocate enough host memory for this tensor
		this->data = (float*)malloc(sizeof(float) * get_size());
	}

	tensor::tensor(vector<size_t> shape, float* data)
	{
		//check the number of dimensions
		if (shape.size() == 0)
			throw exception("Shape must have at least 1 dimension");

		//set the shape to be the input shape
		this->shape = shape;

		//check the size isn't 0
		if (this->get_size() == 0)
			throw exception("Cannot have a tensor with size 0");

		//set the local data pointer to be the input data pointer
		this->data = data;
	}

	tensor tensor::random(size_t size)
	{
		//default the min and max to be 0 and 1
		return random(size, 0, 1);
	}

	tensor tensor::random(size_t size, float min, float max)
	{
		//work out the scale and offset which relate to the max and min
		//values
		float scale = max - min;
		float offset = min;

		//allocate a host pointer with the correct size
		float* data = (float*)malloc(sizeof(float) * size);

		//get a uniformly distributed array of random numbers
		random_host_array(data, scale, offset, size);

		//return a tensor with the data
		return tensor(size, data);
	}

	tensor tensor::random(vector<size_t> shape)
	{
		//default the min and max to be 0 and 1
		return random(shape, 0, 1);
	}

	tensor tensor::random(vector<size_t> shape, float min, float max)
	{
		//work out the scale and offset which relate to the max and min
		//values
		float scale = max - min;
		float offset = min;

		//work out the size manually (as we don't yet have the tensor)
		size_t size = 1;
		for (int i = 0; i < shape.size(); i++)
			size *= shape[i];

		//allocate a host pointer with the correct size
		float* data = (float*)malloc(sizeof(float) * size);

		//get a uniformly distributed array of random numbers
		random_host_array(data, scale, offset, size);

		//return a tensor with the data
		return tensor(shape, data);
	}

	tensor tensor::random_normal(size_t size, float mean, float stddev)
	{
		//allocate a host pointer with the correct size
		float* data = (float*)malloc(sizeof(float) * size);

		//get a normally distributed array of random numbers
		random_normal_array(data, mean, stddev, size);

		//return a tensor with the data
		return tensor(size, data);
	}

	tensor tensor::random_normal(vector<size_t> shape, float mean, float stddev)
	{
		//work out the size manually (as we don't yet have the tensor)
		size_t size = 1;
		for (int i = 0; i < shape.size(); i++)
			size *= shape[i];

		//allocate a host pointer with the correct size
		float* data = (float*)malloc(sizeof(float) * size);

		//get a normally distributed array of random numbers
		random_normal_array(data, mean, stddev, size);

		//return a tensor with the data
		return tensor(shape, data);
	}

	tensor tensor::zeros(size_t size)
	{
		//return a constant value tensor with that value being 0
		return value(size, 0);
	}

	tensor tensor::value(size_t size, float value)
	{
		//allocate a host pointer with the correct size
		float* value_array = (float*)malloc(sizeof(float) * size);

		//fill the host pointer with the desired value
		std::fill(&value_array[0], &value_array[size], value);

		//return a tensor with the data
		return tensor(size, value_array);
	}

	tensor tensor::zeros(vector<size_t> shape)
	{
		//return a constant value tensor with that value being 0
		return value(shape, 0);
	}

	tensor tensor::value(vector<size_t> shape, float value)
	{
		//work out the size manually (as we don't yet have the tensor)
		size_t size = 1;
		for (int i = 0; i < shape.size(); i++)
			size *= shape[i];

		//allocate a host pointer with the correct size
		float* value_array = (float*)malloc(sizeof(float) * size);

		//fill the host pointer with the desired value
		fill(&value_array[0], &value_array[size], value);

		//return a tensor with the data
		return tensor(shape, value_array);
	}

	tensor tensor::one_hot(size_t size, int hot)
	{
		//check the "hot" value is within range
		if (hot >= size)
			throw exception("Hot value out of range");

		//create a tensor of 0s
		tensor o_h = tensor::zeros(size);

		//set the hot value to 1
		o_h.set(hot, 1);

		//return the tensor
		return o_h;
	}

	tensor tensor::stack(vector<tensor> tensors)
	{
		//check that there are tensors to stack
		if (tensors.size() == 0)
			throw exception("Cannot stack 0 tensors");

		//get the size of the first tensor
		size_t t_size = tensors[0].get_size();

		//create an array for the new data for the stacked tensor
		float* new_data = (float*)malloc(sizeof(float) * tensors.size() * t_size);

		//loop throughe each input tensor
		for (int i = 0; i < tensors.size(); i++) {
			//check that this tensor has the right shape
			if (tensors[i].get_size() != t_size)
				throw exception("Cannot stack tensors of different sizes");

			//copy the data from this tensor to the end of the new data buffer
			memcpy(&new_data[t_size * i], tensors[i].get_data(), sizeof(float) * t_size);
		}

		//create the new shape for the new tensor which should have the old
		//shape shifter over by 1
		vector<size_t> new_shape = tensors[0].get_shape();

		//insert the number of tensor to the beginning of the new shape,
		//so the first dimension represents each stacked tensor
		new_shape.insert(new_shape.begin(), tensors.size());

		//return the new tensor with the updated shape and data
		return tensor(new_shape, new_data);
	}

	void tensor::reshape(vector<size_t> new_shape)
	{
		//work out the size of the new shape
		size_t ns_size = 1;
		for (int i = 0; i < new_shape.size(); i++)
			ns_size *= new_shape[i];

		//check the sizes are equal
		if (ns_size != get_size())
			throw exception("Reshaping a tensor cannot change its size");

		//to get here the sizes must be equal so we can safely overwrite the shape
		this->shape = new_shape;
	}

	void tensor::set(int index, float value)
	{
		//check we are not trying to index a multidimensional tensor with just 1 index
		if (shape.size() != 1)
			throw exception("Must use multidimensional index for multidimensional tensor");

		//set the data at the specified index to the new value
		data[index] = value;
	}

	void tensor::set(vector<int> index, float value)
	{
		//get the linear offset for this index in the vector memory
		int linear_offset = get_linear_offset(index);

		//set the data at the specified index to the new value
		data[linear_offset] = value;
	}

	float tensor::get(int index)
	{
		//check we are not trying to index a multidimensional tensor with just 1 index
		if (shape.size() != 1)
			throw exception("Must use multidimensional index for multidimensional tensor");

		//return the data at the specified index
		return data[index];
	}

	float tensor::get(vector<int> index)
	{
		//get the linear offset for this index in the vector memory
		int linear_offset = get_linear_offset(index);

		//return the data at the specified index
		return data[linear_offset];
	}

	void tensor::serialise(char* stream_buffer, size_t offset)
	{
		//get the number of dimensions
		size_t n_dims = shape.size();

		//write the number of dimensions to the start of the buffer so we know how many
		//dimensions to deserialise
		memcpy(&stream_buffer[offset], &n_dims, sizeof(size_t));

		//loop through each dimension in the shape
		for (int dim = 0; dim < n_dims; dim++) {
			//write each dimension to the end of the buffer
			memcpy(&stream_buffer[offset + sizeof(size_t) + dim * sizeof(size_t)], reinterpret_cast<char*>(reinterpret_cast<void*>(&shape[dim])), sizeof(size_t));
		}
		//get the total size of the tensor data
		size_t size = get_size();

		//copy the memory from the device pointer into the end of the buffer
		cuda_safe_call(cudaMemcpy(&stream_buffer[offset + sizeof(size_t) + sizeof(size_t) * n_dims], d_data, sizeof(float) * get_size(), cudaMemcpyDeviceToHost));
	}

	void tensor::deserialise(char* stream_buffer, size_t offset)
	{
		//cast the buffer into a size_t buffer (as we know the first values will be size_t variables)
		size_t* shape_p = reinterpret_cast<size_t*>(reinterpret_cast<void*>(&stream_buffer[offset]));

		//the first size_t will be the number of dimensions (as this was what we serialised first)
		size_t n_dims = shape_p[0];

		//loop through the number of dimesions
		for (int dim = 0; dim < n_dims; dim++) {
			//each consequent size_t represents a dimesion, so we add these to the shape
			shape.push_back(shape_p[dim + 1]);
		}

		//allocate a suitably sized host array for the data
		data = (float*)malloc(sizeof(float) * get_size());

		//copy the remainder of the buffer into the data pointer
		memcpy(data, reinterpret_cast<float*>(reinterpret_cast<void*>(&stream_buffer[offset + sizeof(size_t) + sizeof(size_t) * n_dims])), sizeof(float) * get_size());
	}

	tensor tensor::get_transpose()
	{
		//check that this tensor is a matrix, as the API only supports matrix transposition
		if (shape.size() != 2)
			throw exception("Transpose tensor must have 2 dimensions");

		//check that we are initialised, as this a device operation
		if (!initialised)
			throw exception("Cannot retrieve transpose of undefined matrix");

		//create a new tensor with the shape flipped for the transpose
		tensor * trans = new tensor({ shape[1], shape[0] });

		//allocate the device pointer for this new tensor
		allocate_device_float_pointer(&trans->d_data, get_size());

		//transpose the matrix on the device
		transpose(d_data, trans->d_data, shape[0], shape[1]);

		//set this new tensor to be initialised
		trans->initialised = true;

		//return the new tensor
		return *trans;
	}

	void tensor::initialise()
	{
		//if already initialised, don't re-initialise and return
		if (initialised)
			return;

		//allocate the device memory pointer
		allocate_device_float_pointer(&d_data, get_size());

		//copy the host data into the device
		load_data_into_device(data, d_data, get_size());

		//flag that this tensor is now initialised for device usage
		initialised = true;
	}

	void tensor::uninitialise()
	{
		//if not initialised, abort and return
		if (!initialised)
			return;

		//destroy the pointer to device memory
		deallocate_device_float_pointer(d_data);

		//flag that this tensor is no longer initialised
		initialised = false;
	}

	float* tensor::get_data()
	{
		return data;
	}

	float* tensor::get_dev_pointer()
	{
		return d_data;
	}

	void tensor::set_dev_pointer(float* d_data_p)
	{
		this->d_data = d_data_p;
	}

	size_t tensor::get_dimensions()
	{
		//return the size of the shape vector as this is the number of dimensions
		return shape.size();
	}

	vector<size_t> tensor::get_shape()
	{
		return shape;
	}

	size_t tensor::get_size()
	{
		//set the start size to 1
		size_t size = 1;

		//for each dimesion, mutliply the size to get the 
		//product of each dimension
		for (int i = 0; i < shape.size(); i++)
			size *= shape[i];

		//return the total size
		return size;
	}

	void tensor::set_shape(vector<size_t> shape)
	{
		this->shape = shape;
	}

	int tensor::get_linear_offset(vector<int> index)
	{
		//check that there is an index for the correct number of dimensions
		if (index.size() != shape.size())
			throw exception("Incorrect number of tensor set indices");

		//initialise the offset to 0
		int linear_offset = 0;

		//loop through each dimension
		for (int i = 0; i < shape.size(); i++) {
			//check the index is in the range of its respective dimension
			if (index[i] >= shape[i])
				throw exception("Index out of bound in dimension " + i);

			//find the size of the remaining dimensions
			int dim_off = 1;
			for (int j = i + 1; j < shape.size(); j++) {
				dim_off *= shape[j];
			}

			//add the index of the current dimension to the offset accumulator
			linear_offset += index[i] * dim_off;
		}

		//return the calculated offset
		return linear_offset;
	}
}