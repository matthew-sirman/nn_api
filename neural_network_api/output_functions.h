#pragma once

#include <functional>
#include "instruction_functions.h"

using namespace std;
using namespace nnet::nnet_internal;

namespace nnet {
	namespace nnet_internal {
		//API ENUMERATION
		//Output Function ID
		//Identifier for available output functions for
		//stream serialisation
		enum out_function_id {
			SOFTMAX,
			ARGMAX
		};
	}
	
	namespace instructions {
		//Abstract base class for any network output function
		//Any derived class will be able to be added to the end of 
		//a network model and will be able to be called when the network
		//is run for predictions
		class output_function : public serialisable_function
		{
		public:
			//Default Constructor
			output_function() {}

			//Destructor
			~output_function() {}

			//API FUNCTION
			//Initialise
			//Initialises the function by declaring and allocating memory and initialising
			//the batch size for the function
			virtual void initialise(shape input_shape, size_t batch_size) { __initialise(input_shape, input_shape, batch_size); }

			//API FUNCTION
			//Uninitialise
			//Uninitialises the function by dereferencing memory
			virtual void uninitialise();

			//API FUNCTION
			//Get Out Vector
			//Returns the computed output vector after evaluating
			float* get_out_vector();

			//Get Input Placeholder
			//Return a reference to the input placeholder. This can be used to
			//feed data into this operation
			placeholder& get_input_placeholder() { return input_data_ph; }

		protected:
			//Initialise
			//Internal initaliser specifying input and output shapes separately
			void __initialise(shape input_shape, shape ouput_shape, size_t batch_size);

			//Out Vector
			//The output vector for this specific output function declared on the device
			float* d_out_vector = nullptr;

			//Initialised
			//Flag to indicate if the function has been initialised
			//Defaults to false
			bool initialised = false;

			//Input Data Placeholder
			//Placeholder for the input data for this output function
			placeholder input_data_ph = placeholder("inputs");
		};

		//Softmax Function
		//Output function which applies the softmax function to each output vector in
		//a batch, where the function e^xi/Sum[e^x] is applied over each batch element
		//vector
		class softmax : public output_function {
		public:
			//Default Constructor
			softmax() : softmax(1) {};

			//Constructor specifying the input size
			softmax(size_t input_size);

			//Destructor
			~softmax();

			//API FUNCTION
			//Run
			//Performs softmax over the batch to each element
			void run() override;

			//Function ID
			//Constant set to SOFTMAX to specify that this is a softmax function
			const int func_id = out_function_id::SOFTMAX;
		};

		//Argmax Function
		//Output the maximum argument for each class in the output space. The maximum
		//element represents the chosen class by the network.
		class argmax : public output_function {
		public:
			//Default Constructor
			argmax() : argmax(1) {};

			//Constructor specifying the input size
			argmax(size_t input_size);

			//Destructor
			~argmax();

			//API FUNCTION
			//Run
			//Performs argmax over the batch to each element
			void run() override;

			//API FUNCTION
			//Initialise
			//Initialises the function by declaring and allocating memory and initialising
			//the batch size for the function
			void initialise(shape input_shape, size_t batch_size) override;

			//Function ID
			//Constant set to ARGMAX to specify that this is a argmax function
			const int func_id = out_function_id::ARGMAX;
		};
	}
}