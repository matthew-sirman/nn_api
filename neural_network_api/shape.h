#pragma once

/*
#ifdef NEURALNETWORKAPI_EXPORTS
#define NN_LIB_API __declspec(dllexport)
#else
#define NN_LIB_API __declspec(dllimport)
#endif
*/
#define NN_LIB_API

#include <algorithm>
#include <string>

using namespace std;

struct NN_LIB_API shape {
	size_t width = 1;
	size_t height = 1;
	size_t depth = 1;

	shape() {};
	shape(size_t w) { width = w; }
	shape(size_t w, size_t h) { width = w; height = h; }
	shape(size_t w, size_t h, size_t d) { width = w; height = h; depth = d; }

	const inline size_t size() { return width * height * depth; }
	const inline bool operator==(shape &B) { return (width == B.width && height == B.height && depth == B.depth); }
	const inline bool operator!=(shape &B) { return (width != B.width || height != B.height || depth != B.depth); }
	const inline string str() const { return "(" + to_string(width) + ", " + to_string(height) + ", " + to_string(depth) + ")"; }

	const inline char * serialise() const {
		size_t shape_dims[3] = { width, height, depth };
		return reinterpret_cast<char*>(reinterpret_cast<void*>(shape_dims));
	}

	inline void deserialise(char * stream_buffer, size_t offset) {
		size_t * shape_dims = reinterpret_cast<size_t*>(reinterpret_cast<void*>(&stream_buffer[offset]));
		width = shape_dims[0];
		height = shape_dims[1];
		depth = shape_dims[2];
	}

	inline size_t max_dim() {
		size_t shape_dims[3] = { width, height, depth };
		return *max_element(&shape_dims[0], &shape_dims[2]);
	}
};
