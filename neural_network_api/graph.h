#pragma once

#include <functional>

#include "node.h"
#include "cost_functions.h"
#include "instruction_functions.h"
#include "output_functions.h"

using namespace std;
using namespace nnet::nnet_internal;
using namespace nnet::instructions;
using namespace nnet::cost_functions;

namespace nnet {
	namespace nnet_internal {
		//Graph
		//A generalisable graph class with a template type T. The graph can
		//be traversed forwards and backwards calling a callback function at
		//every node in the graph.
		template <typename T>
		class graph
		{
		public:
			//Default Constructor
			graph() {
				
			}

			//Destructor
			~graph() {

			}

			//API FUNCTION
			//Traverse
			//Traverse through the graph. The specified callback function will be called on every
			//node once each of its parents are activated
			inline virtual void traverse(function<void(node<T>)> callback) {
				//initialise all the nodes to deactivated
				for (auto n : nodes) {
					n->activated = false;
				}
				
				//call the recursive internal traverse method
				__traverse(callback, *entry_point, *exit_point);
			}

			//API FUNCTION
			//Reverse
			//Traverse through the graph but in reverse order (end->start). The specified callback
			//function will be called on every node once each of its children are activated
			inline virtual void reverse(function<void(node<T>)> callback) {
				//initialise all the nodes to deactivated
				for (auto n : nodes) {
					n->activated = false;
				}

				//call the recursive internal reverse method
				__reverse(callback, *exit_point, *entry_point);
			}
			
			//Add Node
			//Add a node object to the graph
			inline virtual void add_node(node<T>* node) {
				nodes.push_back(node);
			}

			//Set Start Point
			//Set the point which the graph will start at
			inline virtual void set_start_point(node<T>* start) {
				this->entry_point = start;
			}

			//Set End Point
			//Set the point which the graph will end at
			inline virtual void set_end_point(node<T>* end) {
				this->exit_point = end;
			}

			//Get Entry Point
			//Get the entry point for this model
			inline virtual node<T> get_entry_node() {
				return *entry_point;
			}

		protected:
			//Entry Point
			//The node the model enters into
			node<T>* entry_point;

			//Nodes
			//List of all the nodes in the model for reference
			vector<node<T>*> nodes;

			//Exit Point
			//The node the model exits from
			node<T>* exit_point;

		private:
			//internal recursive traverse function
			inline void __traverse(function<void(node<T>)> callback, node<T>& start, node<T>& end) {
				//initialise the activated variable for this node to true
				start.activated = true;

				//if any of the parents are deactivated, set this node to
				//deactivated as it cannot propagate yet
				for (node<T>* parent : start.parents) {
					if (!parent->activated)
						start.activated = false;
				}

				//if this node is activated it can be run
				if (start.activated) {
					//throw the callback
					callback(start);

					//if we are at the end of the model, return and start unrolling
					if (start == end)
						return;

					//recursively call the traverse method for each of the children
					for (node<T>* child : start.children) {
						__traverse(callback, *child, end);
					}
				}
			}

			//internal recursive reverse function
			inline void __reverse(function<void(node<T>)> callback, node<T>& start, node<T>& end) {
				//initialise the activated variable for this node to true
				start.activated = true;

				//if any of the children are deactivated, set this node to
				//deactivated as it cannot propagate yet
				for (node<T>* child : start.children) {
					if (!child->activated)
						start.activated = false;
				}

				//if this node is activated it can be run
				if (start.activated) {
					//throw the callback
					callback(start);

					//recursively call the traverse method for each of the parents
					for (node<T>* parent : start.parents) {
						__reverse(callback, *parent, end);
					}
				}
			}
		};
	}

	//Network Graph
	//A specific type of graph storing instruction functions. Used as the backend
	//for the network_model class, and can be used for generalisable neural networks
	//with more freedom but less helper functions.
	class network_graph : public graph<instruction_function*> {
	public:
		//Feed Input Data
		//Feed input data into the start of the graph
		inline void feed_input_data(float* input) {
			entry_point->value->feed_input_data(input);
		}

		//Feed Target Data
		//Feed target data into the cost function for the graph
		inline void feed_target_data(float* targets) {
			cost_func->feed_target_data(targets);
		}

		//Set Network Batch Size
		//Set the batch size for each function in the network
		inline void set_network_batch_size(size_t batch_size) {
			//loop through each node and set the batch size
			for (auto node : nodes) {
				node->value->set_batch_size(batch_size);
			}
		}

		//Set Network Phase
		//Set the current phase for the network graph
		inline void set_network_phase(network_phase phase) {
			//loop through each node and set the phase
			for (auto node : nodes) {
				node->value->set_phase(phase);
			}
		}

		//Set Cost Function
		//Set the cost function for the graph, for use in training.
		inline void set_cost_function(cost_function* cost_func) { this->cost_func = cost_func; }

		//Set Output Function
		//Set the output function for the graph, for use in predicting.
		inline void set_output_function(output_function* output_func) { this->output_func = output_func; }

		//Initialise
		//Initialise each function associated with the graph
		inline void initialise(shape output_shape, size_t batch_size) {
			//initialise each node
			for (auto n : nodes) {
				n->value->initialise(batch_size);
			}

			//initialise the cost if exists
			if (cost_func != nullptr)
				cost_func->initialise(output_shape, batch_size);

			//initialise the output if exists
			if (output_func != nullptr)
				output_func->initialise(output_shape, batch_size);
		}

		//Uninitialise
		//Uninitialise each function associated with the graph
		inline void uninitialise() {
			//uninitialise each node
			for (auto n : nodes) {
				n->value->uninitialise();
			}

			//uninitialise the cost if exists
			if (cost_func != nullptr)
				cost_func->uninitialise();

			//uninitialise the output if exists
			if (output_func != nullptr)
				output_func->uninitialise();
		}

		//Run Cost Derivative
		//Run the derivative of the cost function and feed it into the last node
		//in the graph
		inline void run_cost_derivative(bool get_cost_metric = false) {
			//feed the output from the last node into the cost function
			cost_func->feed_input_data(exit_point->value->get_out_vector());

			//if we are getting the cost, calculate it here
			if (get_cost_metric) {
				cost_func->run();
			}

			//calculate the derivative
			cost_func->cost_derivative();

			//feed the derivative into the last node of the object
			exit_point->value->feed_derivatives(cost_func->get_derivative_vector());
		}

		//Get Input Shape
		//Returns the input shape for the entry to this graph
		inline shape get_input_shape() {
			return entry_point->value->input_shape;
		}

		//Get Output Shape
		//Returns the output shape from the graph, depending on whether it runs
		//through an output function or not
		inline shape get_output_shape() {
			if (output_func != nullptr) {
				return output_func->output_shape;
			}
			else {
				return exit_point->value->output_shape;
			}
		}

		//Get Output
		//Get either the raw output or the output from the specified output function
		inline void get_output(float* output, size_t batch_size) {
			//if the output function exists, feed in data, run it and write it to the outpu
			//otherwise write the raw output to the output
			if (output_func != nullptr) {
				output_func->feed_input_data(exit_point->value->get_out_vector());
				output_func->run();

				cuda_safe_call(cudaMemcpy(output, output_func->get_out_vector(), sizeof(float) * output_func->output_shape.size() * batch_size, cudaMemcpyDeviceToDevice));
			}
			else {
				cuda_safe_call(cudaMemcpy(output, exit_point->value->get_out_vector(), sizeof(float) * exit_point->value->output_shape.size() * batch_size, cudaMemcpyDeviceToDevice));
			}
		}

		//Get Cost Function
		//Returns a pointer to the cost function
		inline cost_function* get_cost_function() {
			return cost_func;
		}

		//Get Train Functions
		//Helper method to return all the trainable functions in the graph
		vector<trainable_function*> get_train_functions() {
			//set up empty list vector
			vector<trainable_function*> t_fs;

			//loop through each node
			for (auto f : nodes) {
				//if the node is trainable add it to the list
				if (f->value->is_train_function())
					t_fs.push_back((trainable_function*)f->value);
			}

			//return the list
			return t_fs;
		}

	private:
		//the cost function for this model if it is training
		cost_function* cost_func = nullptr;

		//the output for this model when making predictions
		output_function* output_func = nullptr;
	};
}