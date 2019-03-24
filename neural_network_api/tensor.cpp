#include "stdafx.h"

#include "tensor.h"

bool tensor::random_initialised = false;
curandGenerator_t tensor::prng;

tensor::tensor(size_t size)
{
	if (size == 0)
		throw exception("Cannot have a tensor with size 0");

	this->shape = { size };
	this->data = (float *)malloc(sizeof(float) * size);
}

tensor::tensor(size_t size, float * data)
{
	if (size == 0)
		throw exception("Cannot have a tensor with size 0");

	this->shape = { size };
	this->data = data;
}

tensor::tensor(vector<size_t> shape)
{
	if (shape.size() == 0)
		throw exception("Shape must have at least 1 dimension");
	this->shape = shape;

	if (this->get_size() == 0)
		throw exception("Cannot have a tensor with size 0");

	this->data = (float *)malloc(sizeof(float) * get_size());
}

tensor::tensor(vector<size_t> shape, float *data)
{
	if (shape.size() == 0)
		throw exception("Shape must have at least 1 dimension");
	this->shape = shape;

	if (this->get_size() == 0)
		throw exception("Cannot have a tensor with size 0");

	this->data = data;
}

tensor::~tensor()
{
}

tensor tensor::random(size_t size)
{
	return random(size, 0, 1);
}

tensor tensor::random(size_t size, float min, float max)
{
	if (!random_initialised) {
		if (TRUE_RAND)
			get_prng(&prng, time(NULL));
		else
			get_prng(&prng, 0);
		random_initialised = true;
	}

	float scale = max - min;
	float offset = min;

	float * data = (float *)malloc(sizeof(float) * size);
	random_host_array(prng, data, scale, offset, size, time(NULL));
	return tensor(size, data);
}

tensor tensor::random(vector<size_t> shape)
{
	return random(shape, 0, 1);
}

tensor tensor::random(vector<size_t> shape, float min, float max)
{
	if (!random_initialised) {
		if (TRUE_RAND)
			get_prng(&prng, time(NULL));
		else
			get_prng(&prng, 0);
		random_initialised = true;
	}

	float scale = max - min;
	float offset = min;

	size_t size = 1;
	for (int i = 0; i < shape.size(); i++)
		size *= shape[i];
	float * data = (float *)malloc(sizeof(float) * size);
	random_host_array(prng, data, scale, offset, size, time(NULL));
	return tensor(shape, data);
}

tensor tensor::zeros(size_t size)
{
	return value(size, 0);
}

tensor tensor::value(size_t size, float value)
{
	float * value_array = (float *)malloc(sizeof(float) * size);
	//fill_array(value_array, value, size);
	std::fill(&value_array[0], &value_array[size], value);
	return tensor(size, value_array);
}

tensor tensor::zeros(vector<size_t> shape)
{
	return value(shape, 0);
}

tensor tensor::value(vector<size_t> shape, float value)
{
	size_t size = 1;
	for (int i = 0; i < shape.size(); i++)
		size *= shape[i];
	float * value_array = (float *)malloc(sizeof(float) * size);
	//fill_array(value_array, value, size);
	fill(&value_array[0], &value_array[size], value);
	return tensor(shape, value_array);
}

tensor tensor::one_hot(size_t size, int hot)
{
	if (hot >= size)
		throw exception("Hot value out of range");

	tensor o_h = tensor::zeros(size);
	o_h.set(hot, 1);

	return o_h;
}

tensor tensor::stack(vector<tensor> tensors)
{
	if (tensors.size() == 0)
		throw exception("Cannot stack 0 tensors");

	size_t t_size = tensors[0].get_size();
	float * new_data = (float *)malloc(sizeof(float) * tensors.size() * t_size);
	for (int i = 0; i < tensors.size(); i++) {
		if (tensors[i].get_size() != t_size)
			throw exception("Cannot stack tensors of different sizes");
		memcpy(&new_data[t_size * i], tensors[i].get_data(), sizeof(float) * t_size);
	}

	vector<size_t> new_shape = tensors[0].get_shape();
	new_shape.insert(new_shape.begin(), tensors.size());

	return tensor(new_shape, new_data);
}

void tensor::reshape(vector<size_t> new_shape)
{
	size_t ns_size = 1;
	for (int i = 0; i < new_shape.size(); i++)
		ns_size *= new_shape[i];
	if (ns_size != get_size())
		throw exception("Reshaping a tensor cannot change its size");

	this->shape = new_shape;
}

void tensor::set(int index, float value)
{
	if (shape.size() != 1)
		throw exception("Must use multidimensional index for multidimensional tensor");

	data[index] = value;
}

void tensor::set(vector<int> index, float value)
{
	int linear_offset = get_linear_offset(index);
	data[linear_offset] = value;
}

float tensor::get(int index)
{
	if (shape.size() != 1)
		throw exception("Must use multidimensional index for multidimensional tensor");

	return data[index];
}

float tensor::get(vector<int> index)
{
	int linear_offset = get_linear_offset(index);
	return data[linear_offset];
}

void tensor::serialise(char * stream_buffer, size_t offset)
{
	size_t n_dims = shape.size();
	memcpy(&stream_buffer[offset], &n_dims, sizeof(size_t));
	for (int dim = 0; dim < n_dims; dim++) {
		memcpy(&stream_buffer[offset + sizeof(size_t) + dim * sizeof(size_t)], reinterpret_cast<char*>(reinterpret_cast<void*>(&shape[dim])), sizeof(size_t));
	}
	size_t size = get_size();
	//memcpy(&stream_buffer[offset + sizeof(size_t) + sizeof(size_t) * n_dims], data, sizeof(float) * get_size());
	cuda_safe_call(cudaMemcpy(&stream_buffer[offset + sizeof(size_t) + sizeof(size_t) * n_dims], d_data, sizeof(float) * get_size(), cudaMemcpyDeviceToHost));
}

void tensor::deserialise(char * stream_buffer, size_t offset)
{
	size_t * shape_p = reinterpret_cast<size_t*>(reinterpret_cast<void*>(&stream_buffer[offset]));
	size_t n_dims = shape_p[0];
	for (int dim = 0; dim < n_dims; dim++) {
		shape.push_back(shape_p[dim + 1]);
	}
	data = (float *)malloc(sizeof(float) * get_size());
	memcpy(data, reinterpret_cast<float*>(reinterpret_cast<void*>(&stream_buffer[offset + sizeof(size_t) + sizeof(size_t) * n_dims])), sizeof(float) * get_size());
}

tensor tensor::get_transpose()
{
	if (shape.size() != 2)
		throw exception("Transpose tensor must have 2 dimensions");
	if (!initialised)
		throw exception("Cannot retrieve transpose of undefined matrix");
	tensor * trans = new tensor({ shape[1], shape[0] });
	allocate_device_pointer(&trans->d_data, get_size());
	transpose(d_data, trans->d_data, shape[0], shape[1]);
	trans->initialised = true;
	return *trans;
}

void tensor::initialise()
{
	if (initialised)
		return;
	allocate_device_pointer(&d_data, get_size());
	load_data_into_device(data, d_data, get_size());
	initialised = true;
}

void tensor::uninitialise()
{
	if (!initialised)
		return;
	deallocate_device_pointer(d_data);
	initialised = false;
}

float * tensor::get_data()
{
	return data;
}

float * tensor::get_dev_pointer()
{
	return d_data;
}

void tensor::set_dev_pointer(float * d_data_p)
{
	this->d_data = d_data_p;
}

size_t tensor::get_dimensions()
{
	return shape.size();
}

vector<size_t> tensor::get_shape()
{
	return shape;
}

size_t tensor::get_size()
{
	size_t size = 1;
	for (int i = 0; i < shape.size(); i++)
		size *= shape[i];
	return size;
}

void tensor::set_shape(vector<size_t> shape)
{
	this->shape = shape;
}

int tensor::get_linear_offset(vector<int> index)
{
	if (index.size() != shape.size())
		throw exception("Incorrect number of tensor set indices");

	int linear_offset = 0;
	for (int i = 0; i < shape.size(); i++) {
		if (index[i] >= shape[i])
			throw exception("Index out of bound in dimension " + i);
		int dim_off = 1;
		for (int j = i + 1; j < shape.size(); j++) {
			dim_off *= shape[j];
		}
		linear_offset += index[i] * dim_off;
	}

	return linear_offset;
}
