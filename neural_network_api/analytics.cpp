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
			printf("Average Loss: %f\n", avg_loss);
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
			printf("Average Loss: %f\n", avg_loss);
		if (log_props & log_property::TIME_STAMP)
			printf("Time elapsed: %fs\n", tmr.elapsed() / 1000.0);
		printf("\n");
		tmr.reset();
	}

	step_tmr.stop();
	if (v == HIGH) {
		printf("step: %d - time: %fms - loss: %f\n", cur_step, step_tmr.elapsed(), avg_loss);
	}

	cur_step += 1;
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
