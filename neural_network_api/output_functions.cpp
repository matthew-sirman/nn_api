#include "stdafx.h"
#include "output_functions.h"

namespace nnet {
	namespace instructions {
		void output_function::__initialise(shape input_shape, shape output_shape, size_t batch_size)
		{
			//allocate a pointer for the output vector
			allocate_device_float_pointer(&d_out_vector, output_shape.size() * batch_size);

			//initialise the batch size and input shape
			this->batch_size = batch_size;
			this->input_shape = input_shape;
			this->output_shape = output_shape;

			//flag that this function is initialised fully
			initialised = true;
		}

		void output_function::uninitialise()
		{
			//destroy the output vector pointer
			deallocate_device_float_pointer(d_out_vector);

			//flag this function is uninitialised
			initialised = false;
		}

		float* output_function::get_out_vector()
		{
			return d_out_vector;
		}

		softmax::softmax(size_t input_size)
		{
			//set the shape to 1d of input size
			this->input_shape = shape(input_size);
		}

		softmax::~softmax()
		{
			//destroy the base function
			output_function::~output_function();
		}

		void softmax::run()
		{
			//apply the softmax probability function over the input space to each element
			apply_softmax(feed_data, d_out_vector, input_shape.width, batch_size, 1);
		}

		argmax::argmax(size_t input_size)
		{
			//set the shape to 1d of input size
			this->input_shape = shape(input_size);
		}

		argmax::~argmax()
		{
			//destroy the base function
			output_function::~output_function();
		}

		void argmax::run()
		{
			//create temporary array for argmax
			int* tmp_argmax;

			cuda_safe_call(cudaMallocManaged(&tmp_argmax, sizeof(int) * input_shape.width * batch_size));

			//apply the softmax probability function over the input space to each element
			apply_argmax(feed_data, tmp_argmax, input_shape.size(), batch_size);

			//cast the int argmax to a float array
			cast_int_to_float(tmp_argmax, d_out_vector, input_shape.width * batch_size);
		}

		void argmax::initialise(shape input_shape, size_t batch_size)
		{
			//call the internal initialiser
			__initialise(input_shape, shape(1), batch_size);
		}
	}
}
