#pragma once

namespace nnet {
	namespace nnet_internal {
		//API ENUMERATION
		//Network Phase
		//The current phase of the network (i.e. training or running)
		enum network_phase {
			PHASE_TRAINING,
			PHASE_RUNNING
		};

		//Operation
		//Base class for any runnable operation. Any class which inherits this
		//class must implement the run method and therefore be runnable.
		class operation
		{
		public:
			//API FUNCTION
			//Abstract Run method
			//Called to propagate through a network
			//Every operation function should be able to be run
			virtual void run() = 0;

			//Set Batch Size
			//Set the current batch size for this operation
			void set_batch_size(size_t batch_size) {
				this->batch_size = batch_size;
			}

			//Feed Input Data
			//Feed data into an operation function for use when running
			virtual void feed_input_data(float* input) {
				this->feed_data = input;
			}

			//Set Phase
			//Informs the function whether the network is training or running
			void set_phase(network_phase phase) {
				this->phase = phase;
			}
		protected:
			//Feed Data
			//The stored feed data for this operation
			float* feed_data = nullptr;

			//Batch Size
			//The preset batch size with which the vectors are initialised
			//Sometimes the batch size for a specific iteration may be smaller
			//however, but it should never be larger
			size_t batch_size = 0;

			//Phase
			//The current phase of the network
			network_phase phase;
		};
	}
}
