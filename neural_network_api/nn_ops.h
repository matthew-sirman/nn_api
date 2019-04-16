#pragma once

#include <functional>

#include "kernel.h"

#include "graph.h"

namespace nnet {
	//Run Metrics
	//Metrics that can be specified for calculating gradients. Metrics
	//specified here will all be calculated at the same time as the gradients
	enum run_metrics {
		METRIC_LOSS = 0x01
	};

	//Run Graph
	//Run a graph with a given input batch and write the result to the specified
	//output batch.
	void run_graph(network_graph g, float* input, float* output, size_t batch_size);

	//Caclulate Gradients
	//Calculate the gradients in a graph given an input batch and expected value batch.
	void calculate_gradients(network_graph g, float* x, float* y, size_t batch_size, int metrics = 0);

	namespace nnet_internal {
		//API FUNCTION
		//Run Graph Callback
		//Callback function called on each node during forward pass traversal
		void __run_graph_callback(node<instruction_function*> node);

		//API FUNCTION
		//Backprop Graph Callback
		//Callback function called on each node during backward pass traversal
		void __backprop_graph_callback(node<instruction_function*> node);
	}
}