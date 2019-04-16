#pragma once

#include "graph.h"

namespace nnet {
	void calculate_gradients(graph g, float* x, float* y, size_t batch_size);
}