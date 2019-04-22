#include "stdafx.h"
#include "nn_ops.h"

namespace nnet {
	void run_graph(network_graph g, float* input, float* output, size_t batch_size)
	{
		//set the current batch size for this run
		g.set_network_batch_size(batch_size);

		//set the phase as "running"
		g.set_network_phase(PHASE_RUNNING);

		//feed the input data into the model
		g.feed_input_data(input);

		//traverse the network calling the specified lambda for each node
		g.traverse([](node<instruction_function*> n) { __run_graph_callback(n); });

		//get the output from the graph and write it into the output variable
		g.get_output(output, batch_size);
	}

	void calculate_gradients(network_graph g, float* x, float* y, size_t batch_size, int metrics)
	{
		//set the current batch size for calculating these gradients
		g.set_network_batch_size(batch_size);

		//set the phase as "training"
		g.set_network_phase(PHASE_TRAINING);

		//feed the input data into the model
		g.feed_input_data(x);

		//feed the target data into the model
		g.feed_target_data(y);

		//traverse the network calling the specified lambda for each node
		g.traverse([](node<instruction_function*> n) { __run_graph_callback(n); });

		//get the derivative of the cost function and calculate the loss if the metric flag
		//instructs to
		g.run_cost_derivative(metrics & METRIC_LOSS);

		//backpropagate through the network graph
		g.reverse([](node<instruction_function*> n) { __backprop_graph_callback(n); });
	}
}

void nnet::nnet_internal::__run_graph_callback(node<instruction_function*> node)
{
	//run the node
	node.value->run();

	//feed the result into each child of this node
	for (auto child : node.children) {
		child->value->feed(child->value->get_input_placeholder(), node.value->get_out_vector());
	}
}

void nnet::nnet_internal::__backprop_graph_callback(node<instruction_function*> node)
{
	//backpropagate the node
	node.value->back_propagate();

	//feed the result into each parents of this node
	for (auto parent : node.parents) {
		parent->value->feed(parent->value->get_derivative_placeholder(), node.value->get_derivative_vector());
	}
}
