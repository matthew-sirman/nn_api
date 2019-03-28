#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#include <curand_kernel.h>

#include <vector>
#include "instructions_kernel.h"

using namespace std;

constexpr bool TRUE_RAND = true;

class NN_LIB_API tensor
{
public:
	tensor() {};
	tensor(size_t size);
	tensor(size_t size, float *data);
	tensor(vector<size_t> shape);
	tensor(vector<size_t> shape, float *data);
	~tensor();

	static tensor random(size_t size);
	static tensor random(size_t size, float min, float max);
	static tensor random(vector<size_t> shape);
	static tensor random(vector<size_t> shape, float min, float max);
	
	static tensor random_normal(size_t size, float mean = 0.0f, float stddev = 1.0f);
	static tensor random_normal(vector<size_t> shape, float mean = 0.0f, float stddev = 1.0f);

	static tensor zeros(size_t size);
	static tensor value(size_t size, float value);
	static tensor zeros(vector<size_t> shape);
	static tensor value(vector<size_t> shape, float value);
	static tensor one_hot(size_t size, int hot);

	static tensor stack(vector<tensor> tensors);

	void reshape(vector<size_t> new_shape);

	void set(int index, float value);
	void set(vector<int> index, float value);
	float get(int index);
	float get(vector<int> index);

	void serialise(char * stream_buffer, size_t offset);
	void deserialise(char * stream_buffer, size_t offset);

	tensor get_transpose();

	void initialise();
	void uninitialise();

	float *get_data();

	float *get_dev_pointer();
	void set_dev_pointer(float * d_data_p);

	size_t get_dimensions();
	vector<size_t> get_shape();
	size_t get_size();

	void set_shape(vector<size_t> shape);

private:
	int get_linear_offset(vector<int> index);
	float *data;
	float *d_data;

	vector<size_t> shape;

	bool initialised = false;

	static bool random_initialised;
	static curandGenerator_t prng;
};

