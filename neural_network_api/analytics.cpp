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
	log_props |= prop;
}

void analytics::drop_log(log_property prop)
{
	log_props &= ~prop;
}

/*void analytics::start_log()
{
	tmr.reset();
	tmr.start();
}

void analytics::stop_log()
{
	tmr.stop();
}*/

void analytics::init_logging()
{
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
	cur_epoch = 0;
	cur_step = 0;
}

void analytics::end_logging()
{
	printf("Done\n");
}

void analytics::on_epoch_start()
{
	if (log_type == PER_EPOCH && v != LOW) {
		printf("Epoch %d starting...\n", cur_epoch + 1);
		//printf("\n");
		tmr.start();
	}
}

void analytics::on_step_start()
{
	step_tmr.start();
	if (log_type == PER_STEPS && cur_step % log_steps == 0) {
		tmr.start();
	}
}

void analytics::on_epoch_end(float avg_loss)
{
	if (log_type == log_output_type::PER_EPOCH && v != LOW) {
		tmr.stop();
		if (log_props & log_property::STEP)
			printf("Epoch %d complete\n", cur_epoch + 1);
		if (log_props & log_property::AVG_LOSS)
			printf("Average Loss: %e\n", avg_loss);
		if (log_props & log_property::TIME_STAMP)
			printf("Time elapsed: %fs\n", tmr.elapsed() / 1000.0);
		printf("\n");
		tmr.reset();
	}
	cur_epoch += 1;
}

void analytics::on_step_end(float avg_loss)
{
	if (log_type == log_output_type::PER_STEPS && v != LOW && cur_step % log_steps == log_steps - 1) {
		tmr.stop();
		if (log_props & log_property::STEP)
			printf("Step %d complete\n", cur_step + 1);
		if (log_props & log_property::AVG_LOSS)
			printf("Average Loss: %e\n", avg_loss);
		if (log_props & log_property::TIME_STAMP)
			printf("Time elapsed: %fs\n", tmr.elapsed() / 1000.0);
		printf("\n");
		tmr.reset();
	}

	step_tmr.stop();
	if (v == HIGH) {
		printf("step: %5d - time: %1.4fms - loss: %.5e - acc: %.4f\n", cur_step, step_tmr.elapsed(), avg_loss, exp(-avg_loss));
		max_avg_cost = max(avg_loss, max_avg_cost);
		if (plotting) {
			plot_data.emplace_back(cur_step, avg_loss);
			int start = 0;
			if (cur_step > GRAPH_PLOT_STEP_COUNT)
				start = cur_step - GRAPH_PLOT_STEP_COUNT;
			//*gp << "plot [" << to_string(start) << ":" << to_string(start + GRAPH_PLOT_STEP_COUNT) << "] [0:" << to_string(max_avg_cost) << "] '-' with lines title 'Cost'\n";
			//gp->send1d(data);
			//gp->flush();
			fprintf(plot_str, "plot [%d:%d] [0:*] '-' with lines title 'Cost'\n", start, start + GRAPH_PLOT_STEP_COUNT);
			for (int plt = start; plt < start + GRAPH_PLOT_STEP_COUNT; plt++) {
				if (plt < plot_data.size()) {
					fprintf(plot_str, "%f %f\n", plot_data[plt].first, plot_data[plt].second);
				}
			}
			fprintf(plot_str, "e\n");
			fflush(plot_str);
		}
	}

	cur_step += 1;
}

void analytics::plot()
{
	//begin live graph plot
	
	plot_str = _popen("gnuplot --persist", "w");

	fprintf(plot_str, "set title 'Analytics Graphing'\n");
	fprintf(plot_str, "set xlabel 'Step'\n");
	fprintf(plot_str, "show xlabel\n");
	fprintf(plot_str, "set ylabel 'Cost'\n");
	fprintf(plot_str, "show ylabel\n");
	fprintf(plot_str, "set grid ytics lc rgb '#bbbbbb' lw 1 lt 0\n");
	fprintf(plot_str, "set grid xtics lc rgb '#bbbbbb' lw 1 lt 0\n");

	/*gp = new Gnuplot();
	*gp << "--persist\n";
	*gp << "set title 'Analytics Graphing'\n";
	*gp << "set xlabel 'Step'\n";
	*gp << "show xlabel\n";
	*gp << "set ylabel 'Cost'\n";
	*gp << "show ylabel\n";*/
	//*gp << "plot '-' with lines title 'Cost'\n";
	//gp->send1d(data);

	plotting = true;
}

/*void analytics::print_log(float avg_loss)
{
	if (!verbose)
		return;
	double elapsed = tmr.elapsed();
	if (log_props & log_property::STEP)
		printf("Epoch: %d\n", cur_epoch);
	if (log_props & log_property::AVG_LOSS)
		printf("Loss: %f\n", avg_loss);
	if (log_props & log_property::TIME_STAMP)
		printf("Time elapsed: %f\n", elapsed);
	printf("\n");
}*/
