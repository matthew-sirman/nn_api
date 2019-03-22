#pragma once

#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif

#include "batch_iterator.h"

class NN_LIB_API heirarchy_image_loader : public batch_iterator
{
public:
	heirarchy_image_loader();
	~heirarchy_image_loader();
};

