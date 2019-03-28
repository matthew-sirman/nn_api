#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#include <string>
#include <fstream>

#include <boost\tuple\tuple.hpp>

#include "gnuplot-iostream.h"

#include "timer.h"

constexpr auto GRAPH_PLOT_STEP_COUNT = 100;

using namespace std;

enum NN_LIB_API log_output_type {
	PER_EPOCH = 0x01,
	PER_STEPS = 0x02
};

enum NN_LIB_API log_property {
	TIME_STAMP = 0x01,
	AVG_LOSS = 0x02,
	STEP = 0x04
};

enum NN_LIB_API verbosity {
	LOW,
	MEDIUM,
	HIGH
};

class NN_LIB_API analytics
{
public:
	analytics();
	analytics(log_output_type out_type);
	analytics(int steps);
	analytics(log_output_type out_type, verbosity v) {
		this->log_type = out_type;
		this->v = v;
	}
	~analytics();

	void add_log(log_property prop);
	void drop_log(log_property prop);

	void init_logging();
	void end_logging();

	void on_epoch_start();
	void on_step_start();

	void on_epoch_end(float avg_loss = 0);
	void on_step_end(float avg_loss = 0);

	void plot();

private:
	//Gnuplot * gp;
	FILE * plot_str;
	std::vector<std::pair<double, double>> plot_data;

	int log_type = log_output_type::PER_EPOCH;
	int log_props = log_property::STEP | log_property::TIME_STAMP | log_property::AVG_LOSS;

	verbosity v = verbosity::MEDIUM;

	float elapsed_time;

	int log_steps;

	int cur_epoch = 0;
	int cur_step = 0;

	timer tmr;
	timer step_tmr;

	bool plotting = false;
	float max_avg_cost = 0;
};

