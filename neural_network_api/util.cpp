#include "stdafx.h"

#include "util.h"

namespace nnet {
	namespace util {
		int ceil_div(int divisor, int number)
		{
			//returns the rounded up integer of the real quotient between the number
			//and divisor
			return ceil((float)number / divisor);
		}
	}
}