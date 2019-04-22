#pragma once

#include <math.h>

namespace nnet {
	namespace util {
		//Ceil Divide
		//Returns the number divided by the divisor, but rounds up if
		//the result is fractional
		int ceil_div(int divisor, int number)
		{
			//returns the rounded up integer of the real quotient between the number
			//and divisor
			return ceil((float)number / divisor);
		}
	}
}