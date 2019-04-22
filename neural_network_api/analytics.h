#pragma once

#include <string>
#include <fstream>

#include <boost\tuple\tuple.hpp>
#include "gnuplot-iostream.h"

#include "timer.h"

constexpr auto GRAPH_PLOT_STEP_COUNT = 100;

using namespace std;

namespace nnet {
	//logging option enumerations

	//Log Output Type
	//Specify when during training the logger should output
	enum log_output_type {
		LOG_PER_EPOCH = 0x01,
		LOG_PER_STEPS = 0x02
	};

	//this uses a flag system, so different properties can be superimposed with bitwise ops

	//Log Property
	//Specify which properties should be logged by the logger
	enum log_property {
		LOG_TIME_STAMP = 0x01,
		LOG_AVG_LOSS = 0x02,
		LOG_STEP = 0x04
	};


	//Verbosity
	//Specify the verbosity with which the logger outputs information
	enum verbosity {
		VERBOSITY_LOW,
		VERBOSITY_MEDIUM,
		VERBOSITY_HIGH
	};

	//Analytics Logging Class
	//Create a logger to print to the console periodically with update information to show how
	//well training is going. Properties such as loss and time help to indicate how effectively
	//the model is learning and how quickly it is training.
	//For realtime graph plotting (average batch loss against training step) use the plot() method
	class analytics
	{
	public:
		//constructors / destructors

		//Default Constructor
		//Logging type will default to PER_EPOCH logging 
		//Verbosity will default to MEDIUM
		analytics();

		//Output Type Contructor
		//Verbosity will default to MEDIUM
		analytics(log_output_type out_type);

		//Step count Constructor
		//PER_STEP logging type with a log output every specified number of steps
		analytics(int steps);

		//Output Type and Verbosity Type constructor
		//Possible output types are: PER_EPOCH, PER_STEP
		//Possible verbosities are: LOW, MEDIUM, HIGH
		analytics(log_output_type out_type, verbosity v) {
			this->log_type = out_type;
			this->v = v;
			plot_data = vector<pair<double, double>>();
		}

		//Destructor
		~analytics();

		//public functions for setting up logger

		//Add a logging property to the logger
		//Logging properties are: TIME_STAMP, AVG_LOSS, STEP
		void add_log(log_property prop);

		//Remove a logging property from the logger
		void drop_log(log_property prop);

		//API FUNCTION
		//Begin the logging for this logger
		void init_logging();

		//API FUNCTION
		//Terminate the logging for this logger
		void end_logging();

		//"event" functions called by the network_model class

		//API FUNCTION
		//Event callback on the start of each epoch
		void on_epoch_start();

		//API FUNCTION
		//Event callback on the start of each step
		void on_step_start();

		//API FUNCTION
		//Event callback on the end of each epoch
		void on_epoch_end(float avg_loss = 0);

		//API FUNCTION
		//Event callback on the end of each step
		void on_step_end(float avg_loss = 0);

		//Plot a realtime graph of the step loss against time whilst training
		void plot();

	private:
		//plot stream (GNU-Plot uses a stream to write to the graph like a console)
		FILE* plot_str;
		//data points for 
		std::vector<std::pair<double, double>> plot_data;

		//variables to hold the log type and the logging properties for the logger. Defaults
		//to PER_EPOCH logging, all available logging properties and medium verbosity
		int log_type = log_output_type::LOG_PER_EPOCH;
		int log_props = log_property::LOG_STEP | log_property::LOG_TIME_STAMP | log_property::LOG_AVG_LOSS;

		verbosity v = verbosity::VERBOSITY_MEDIUM;

		//simple variable to hold the elapsed time for each logging step (epoch/step)
		float elapsed_time;

		//period to output log if PER_STEP specified
		int log_steps;

		//holds the current step and epoch the training is on
		int cur_epoch = 0;
		int cur_step = 0;

		//a timer to hold both the time for each logging period, and each individual step
		timer tmr;
		timer step_tmr;

		//did the user specify wanting to plot the graph?
		bool plotting = false;
	};
}