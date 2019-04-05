#include "stdafx.h"

#include "analytics.h"


analytics::analytics()
	: analytics(log_output_type::PER_EPOCH)
{

}

analytics::analytics(log_output_type out_type)
{
	log_type = out_type;
	tmr = timer();
}

analytics::analytics(int steps)
{
	log_type = log_output_type::PER_STEPS;
	tmr = timer();
	this->log_steps = steps;
}


analytics::~analytics()
{
}

void analytics::add_log(log_property prop)
{
	//bitwise or operation to add this property to the flag
	log_props |= prop;
}

void analytics::drop_log(log_property prop)
{
	//bitwise and with not prop to remove property from the flag
	log_props &= ~prop;
}

void analytics::init_logging()
{
	//indicate which verbosity level will be used by switching the verbosity
	switch (v) {
	case HIGH:
		printf("Analytics logging verbosity: high\n");
		break;
	case MEDIUM:
		printf("Analytics logging verbosity: medium\n");
		break;
	case LOW:
		printf("Analytics logging verbosity: low\n");
		break;
	}
	//initialise local vars
	cur_epoch = 0;
	cur_step = 0;
}

void analytics::end_logging()
{
	//indication that training is finished (called by the network_model upon completion)
	printf("Done\n");
}

void analytics::on_epoch_start()
{
	//print if the logging type is PER_EPOCH, unless low verbosity specified
	if (log_type == PER_EPOCH && v != LOW) {
		//add 1 to cur_epoch so it counts from 1 rather than from 0
		printf("Epoch %d starting...\n", cur_epoch + 1);
		tmr.start();
	}
}

void analytics::on_step_start()
{
	//set the dedicated step timer to begin
	step_tmr.start();
	//if it is per steps logging and the current step is a multiple of the step period,
	//begin the period timer
	if (log_type == PER_STEPS && cur_step % log_steps == 0) {
		tmr.start();
	}
}

void analytics::on_epoch_end(float avg_loss)
{
	//if the logging is on epoch and verbosity isn't low
	if (log_type == log_output_type::PER_EPOCH && v != LOW) {
		//stop the period timer
		tmr.stop();

		//print each property if it is specified within the flag
		if (log_props & log_property::STEP)
			printf("Epoch %d complete\n", cur_epoch + 1);
		if (log_props & log_property::AVG_LOSS)
			printf("Average Loss: %e\n", avg_loss);
		if (log_props & log_property::TIME_STAMP)
			printf("Time elapsed: %fs\n", tmr.elapsed() / 1000.0);
		printf("\n");
	}
	//increment the epoch
	cur_epoch += 1;
}

void analytics::on_step_end(float avg_loss)
{
	//if the logging is on step and verbosity isn't low and it is a suitable step to output
	if (log_type == log_output_type::PER_STEPS && v != LOW && cur_step % log_steps == log_steps - 1) {
		//stop the period timer
		tmr.stop();

		//print each property if it is specified within the flag
		if (log_props & log_property::STEP)
			printf("Step %d complete\n", cur_step + 1);
		if (log_props & log_property::AVG_LOSS)
			printf("Average Loss: %e\n", avg_loss);
		if (log_props & log_property::TIME_STAMP)
			printf("Time elapsed: %fs\n", tmr.elapsed() / 1000.0);
		printf("\n");
	}

	//stop the step timer
	step_tmr.stop();

	//if the verbosity is high
	if (v == HIGH) {
		//print a per step line stating the step loss and approximate accuracy
		printf("step: %5d - time: %1.4fms - loss: %.5e - acc: %.4f\n", cur_step, step_tmr.elapsed(), avg_loss, exp(-avg_loss));
	}

	//if the user specified plotting
	if (plotting) {
		//add the current step and cost to the end of the plot_data vector
		plot_data.emplace_back(cur_step, avg_loss);

		//default the start point to 0
		int start = 0;

		//if more points than the width of the graph have been plotted, update the start point
		if (cur_step > GRAPH_PLOT_STEP_COUNT)
			start = cur_step - GRAPH_PLOT_STEP_COUNT;

		//write to the GNU-Plot stream to plot the graph
		fprintf(plot_str, "plot [%d:%d] [0:*] '-' with lines title 'Cost'\n", start, start + GRAPH_PLOT_STEP_COUNT);

		//iterate through last GRAPH_PLOT_STEP_COUNT steps and write all to the stream
		for (int plt = start; plt < start + GRAPH_PLOT_STEP_COUNT; plt++) {
			//check that the plot is within range (important for before GRAPH_PLOT_STEP_COUNT steps have been reached)
			if (plt < plot_data.size()) {
				//write point to stream
				fprintf(plot_str, "%f %f\n", plot_data[plt].first, plot_data[plt].second);
			}
		}
		//write end flag to stream
		fprintf(plot_str, "e\n");

		//flush the stream (update the graph)
		fflush(plot_str);
	}

	//increment the step counter
	cur_step += 1;
}

void analytics::plot()
{
	//begin live graph plot
	plot_str = _popen("gnuplot --persist", "w");

	//setup GNU-Plot parameters (title, axis labels, background grids)
	fprintf(plot_str, "set title 'Analytics Graphing'\n");
	fprintf(plot_str, "set xlabel 'Step'\n");
	fprintf(plot_str, "show xlabel\n");
	fprintf(plot_str, "set ylabel 'Cost'\n");
	fprintf(plot_str, "show ylabel\n");
	fprintf(plot_str, "set grid ytics lc rgb '#bbbbbb' lw 1 lt 0\n");
	fprintf(plot_str, "set grid xtics lc rgb '#bbbbbb' lw 1 lt 0\n");
	fflush(plot_str);

	//set the plotting flag to true to indicate that this logger should plot
	plotting = true;
}
