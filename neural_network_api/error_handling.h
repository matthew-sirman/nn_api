#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>

//ERR_ASSERT
//Macro to assert an error message given an expression
#define ERR_ASSERT(expr, err) if (!!(expr)) { \
	std::cout << "Failed on line " << __LINE__ << " of file " << __FILE__ << " in function \"" << __func__ << "\" with error: " << err << endl; \
	exit(-1); \
}