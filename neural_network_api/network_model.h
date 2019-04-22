#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <Windows.h>
#include <codecvt>
#include <time.h>

#include "nn_ops.h"
#include "cost_functions.h"
#include "optimisers.h"
#include "analytics.h"
#include "batch_iterator.h"
#include "variable_initialiser.h"
#include "layers.h"
#include "output_functions.h"
#include "timer.h"

#include "error_handling.h"

using namespace std;

using namespace nnet::nnet_internal;
using namespace nnet::instructions;
using namespace nnet::optimisers;
using namespace nnet::cost_functions;

namespace nnet {

	//Checkpoint
	//Use for checkpointed training of a model. The model will save a checkpoint after every
	//period of training defined to avoid losing the model in the event of a crash etc.
	//CHECKPOINT_NONE - disable checkpointing
	//CHECKPOINT_PER_EPOCH - save a checkpoint at the end of every epoch
	//CHECKPOINT_PER_STEPS - save a checkpoint after every given number of steps
	enum checkpoint_type {
		CHECKPOINT_NONE = 0x00,
		CHECKPOINT_PER_EPOCH = 0x01,
		CHECKPOINT_PER_STEPS = 0x02,
		CHECKPOINT_END = 0x04
	};

	//Network Model
	//Defines a trainable neural network model. There are numerous helper functions to
	//more easily add layers to the network model. Once a model is defined, optimisers,
	//cost functions and output functions can all be defined. The network can then be trained
	//with the train() function.
	//Trained models can be written to files, and then reloaded from files.
	//Trained models can be evaluated (to get a percentage accuracy) and run (to give predictions
	//for a tensor of stimuli)
	class network_model
	{
	public:
		//Default constructor for the network model
		network_model() {};

		//Destructor
		~network_model() {};

		//Entry
		//Specify the entry input shape to the neural network model.
		//All data passed through must have this shape.
		void entry(shape entry_shape);

		//Add
		//Adds a trainable "add" function to the end of the network model.
		//The "biases" tensor will be added to the input vector when run.
		//The "biases" tensor will be trained, unless locked.
		//Returns a pointer to the added function.
		add_function* add(tensor biases);

		//Matmul
		//Adds a trainable "matmul" function to the end of the network model.
		//The "weights" tensor will be multiplied with the input vector when run.
		//The "weights" tensor will be trained, unless locked.
		//Returns a pointer to the added function.
		matmul_function* matmul(tensor weights);

		//Dense
		//Adds a dense layer to the end of the network model.
		//A dense layer consists of a single matmul layer and a single
		//add layer. The dense layer will create tensors for weights and
		//biases automatically, which will be trained.
		//Returns a reference to the added layer.
		dense_layer dense(size_t units, variable_initialiser weight_initialiser = variable_initialiser(), variable_initialiser bias_initialiser = variable_initialiser());

		//Conv2D
		//Adds a trainable 2D convolutional filter function to the end of the network model.
		//A set of n filters with the specified shape will be automatically
		//created and trained.
		//Padding can either be VALID or SAME.
		//VALID padding will reduce the size (width by height) of the output map depending on the 
		//filter shape.
		//SAME padding will keep the size (width by height) of the output map the same as the input
		//map.
		//Returns a pointer to the added function.
		conv2d_function* conv2d(shape filter_shape, size_t n_filters, padding_type padding = padding_type::PADDING_VALID, variable_initialiser initialiser = variable_initialiser());

		//Conv2D
		//Adds a trainable 2D convolutional filter function to the end of the network model.
		//A set of n filters with the specified shape will be automatically
		//created and trained.
		//Padding size can be explicitly set. The padding size will create a
		//border of 0s, where the border thickness is dependant on the width and height 
		//of the padding shape.
		//Returns a pointer to the added function.
		conv2d_function* conv2d(shape filter_shape, size_t n_filters, shape padding, variable_initialiser initialiser = variable_initialiser());

		//Conv2D
		//Adds a trainable 2D convolutional filter function to the end of the network model.
		//A 4D tensor of filters must be supplied.
		//Padding size can be explicitly set. The padding size will create a
		//border of 0s, where the border thickness is dependant on the width and height 
		//of the padding shape.
		//The "biases" tensor will be trained, unless locked.
		//Returns a pointer to the added function.
		conv2d_function* conv2d(tensor filter, shape padding);

		//Max Pool
		//Adds a max pooling function to the end of the network model.
		//Strides over the input space and returns the maximum value in each pool
		//defined by the pool_size parameter.
		//Returns a pointer to the added function.
		max_pool_function* max_pool(shape pool_size, shape stride);

		//Flatten
		//Adds a flatten function to the end of the network model.
		//Flatten will take a multidimensional input and flatten it into a single dimension
		//with the same total size.
		//Returns a pointer to the added function.
		flatten_function* flatten();

		//Reshape
		//Adds a reshape function to the end of the network model.
		//Reshape will take an input with a given shape and transform it into a new specified
		//output shape with the same total size.
		//Returns a pointer to the added function.
		reshape_function* reshape(shape output_shape);

		//Dropout
		//Adds a dropout function to the end of the network model.
		//Dropout will randomly drop nodes with a given probability for regularisation.
		//Returns a pointer to the added function.
		dropout_function* dropout(float keep_rate);

		//ReLU
		//Adds a ReLU activation function to the end of the network model.
		//ReLU will apply the function f(x) = max(0, x) elementwise over the entire input.
		//Returns a pointer to the added function.
		relu_function* relu();

		//Leaky ReLU
		//Adds a Leaky ReLU activation function to the end of the network model.
		//Leaky ReLU will apply the function f(x) = {x: x > 0, alpha: x <= 0} elementwise
		//over the entire input.
		//Returns a pointer to the added function.
		leaky_relu_function* leaky_relu(float alpha);

		//Tanh
		//Adds a Tanh activation function to the end of the network model.
		//Tanh will apply the function f(x) = tanh(x) elementwise over the entire input.
		//Returns a pointer to the added function.
		tanh_function* tanh();

		//Sigmoid
		//Adds a Sigmoid activation function to the end of the network model.
		//Sigmoid will apply the function f(x) = 1/(1 + e^(-x)) elementwise over the entire input.
		//Returns a pointer to the added function.
		sigmoid_function* sigmoid();

		//Function
		//Adds a general function to the end of the network model.
		//This allows for user defined functions to be added to the model,
		//provided they implement the abstract "instruction_function" class.
		//NOTE: In current version, custom functions can not be serialised and
		//saved
		//Returns a pointer to the added function.
		instruction_function* function(instruction_function* func);

		//Set Cost Function
		//Set the cost function for the network, for use in training.
		//Template type T must inherit from the "cost_function" class.
		//Instantiation of the cost function is handled automatically, only
		//the type is required.
		//Returns a pointer to the added function.
		template <typename T>
		inline cost_function* set_cost_function() { 
			cost_function* cost_func = new T(); 
			model_graph.set_cost_function(cost_func);
			return cost_func;
		}

		//Set Output Function
		//Set the output function for the network, for use in predicting.
		//Template type T must inherit from the "output_function" class.
		//Instantiation of the output function is handled automatically, only
		//the type is required.
		//Returns a pointer to the added function.
		template <typename T>
		inline output_function* set_output_function() { 
			output_function* output_func = new T(); 
			model_graph.set_output_function(output_func);
			return output_func; 
		}

		//Set Optimiser
		//Set the optimiser for the network, for use in training.
		//The optimiser must be instantiated manually, so parameters
		//e.g. learning rate can be set on the object before passing through.
		inline void set_optimiser(optimiser * opt) { this->opt = opt; }

		//Add Logger
		//Add an Analytics logger to the model. A logger will print updates into the
		//console regarding current training progress, step time and loss. A logger 
		//may also be used to plot a realtime graph of cost against training step.
		//To display a realtime graph, call the plot() method on the logger object.
		void add_logger(analytics& logger);

		//Initialise Model
		//Initialise the network model before any network operation can be performed on it.
		//A batch size is specified here to set up each function in the model and allocate
		//appropriately sized regions of memory.
		//Initialise model MUST be called before training, evaluating or predicting.
		void initialise_model(size_t batch_size = 128);

		//Uninitialise Model
		//Uninitialise the network model by freeing memory and closing the model down. After
		//uninitialising, the model will no longer be able to handle any actions such as training
		//evaluating or predicting. The uninitialise function will automatically handle all
		//uninitialisation within the functions held in the model. This function should be called
		//when use of the model is finished.
		void uninitialise_model();

		//Predict
		//Run a tensor of inputs (where the tensor shape should be each input in the first
		//dimension, and the shape of each input in the following dimensions)
		//Returns a tensor of predictions which will be formatted depending on the output function.
		tensor predict(tensor input);

		//Train
		//Train the network model by minimising the loss between expected and observed results
		//using the optimiser specified.
		//Each input in train_x will be propagated through the network to give an observed result,
		//which will be compared to the expected result in train_y. The train function will iteratively
		//update the trainable parameters in the network model to give better predictions over a number
		//of training steps (epochs).
		//The train_x tensor should have each element in the first dimension, followed by the information about
		//that element in the following dimensions.
		//The train_y tensor should have each element in the first dimension, followed by the expected result for
		//that element in the following dimensions.
		void train(tensor train_x, tensor train_y, int epochs, bool from_start = false);

		//Train
		//Train the network model by minimising the loss between expected and observed results
		//using the optimiser specified.
		//Each input from the iterator will be propagated through the network to give an observed result,
		//which will be compared to the expected result from the iterator. The train function will iteratively
		//update the trainable parameters in the network model to give better predictions over a number
		//of training steps (epochs).
		void train(batch_iterator &b_iter, int epochs, bool from_start = false);

		//Evaluate
		//Evaluate the network model by testing what percentage of inputs from a separate unseen dataset 
		//it can correctly classify.
		//The test_x tensor should have each element in the first dimension, followed by the information about
		//that element in the following dimensions.
		//The test_y tensor should have each element in the first dimension, followed by the expected result for
		//that element in the following dimensions.
		//Returns a number between 0 and 1, where 0 represents 0% accuracy, and 1 represents 100% accuracy
		//over the test set.
		float evaluate(tensor test_x, tensor test_y);

		//Evaluate
		//Evaluate the network model by testing what percentage of inputs from a separate unseen dataset 
		//it can correctly classify.
		//Each input from the batch iterator will be compared to the output from the batch iterator to give
		//a percentage accuracy.
		//Returns a number between 0 and 1, where 0 represents 0% accuracy, and 1 represents 100% accuracy
		//over the test set.
		float evaluate(batch_iterator &b_iter);

		//Export Model To File
		//Export a model to a specified folder given a name.
		//The model will be stored in a folder with the given name, and a .model file will be created.
		//All trainable parameters will be saved for future use.
		void export_model_to_file(string model_folder, string model_name);

		//Import Model From File
		//Statically load a model from the specified folder.
		//Returns a network model from the .model file located in the model subfolder.
		//All trainable parameters are loaded back into the network functions.
		//The model must still be initialised before use.
		static network_model import_model_from_file(string model_folder, string model_name);

		//Set Checkpoint
		//Setup checkpointing if this model should be periodically saved. Specify the period and
		//save location for this model.
		void set_checkpoint(int c_type, string model_folder, string model_name, int steps = 0);
	private:
		//internal function to write a model to a file
		void __export_model_to_file(string model_folder, string model_name);

		//the internal graph to store this network under.
		//the graph will be sequential if create by this model class
		sequential_network_graph model_graph;
		
		//holds the current layer shape during model definition
		shape init_layer_shape;

		//the optimiser function for this model if it is training
		optimiser * opt = nullptr;

		//the operation to run the optimiser
		optimiser_operation* opt_operation = nullptr;

		//the logger for training outputs and graph plotting
		analytics *analytics_logger = nullptr;

		//the shape with which the model will output predictions, e.g. number of classes
		shape output_shape;

		//the size of each batch run through the model
		size_t batch_size = 128;

		//the size of the largest layer in the model (for memory allocations)
		//size_t largest_layer_size = 0;

		//flag to determine whether the model is initialised or not
		bool model_initialised = false;

		//flag to determine whether the entry shape of the model has been specified
		bool __ent_spec = false;

		//the path to the folder this model should be saved in
		string file_path;

		//the name of the model for checkpointing
		string model_name;

		//the period which the model is saved
		int cpt = checkpoint_type::CHECKPOINT_NONE;

		//if step checkpointing is chosen, this is the number of steps it checkpoints after
		int cpt_steps = 0;

		//number of iterations this model has made
		int __step = 0;
	};
}

