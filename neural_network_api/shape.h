#pragma once

#include <algorithm>
#include <string>

using namespace std;

namespace nnet {
	//Shape
	//A structure to represent a 3D shape with a width, height and depth.
	struct shape {
		//Width
		//The width of this shape
		size_t width = 1;

		//Height
		//The height of this shape
		size_t height = 1;

		//Depth
		//The depth of this shape
		size_t depth = 1;

		//Default Constructor
		//Defaults to (1, 1, 1)
		shape() {};

		//Constructor specifying width
		//Shape: (w, 1, 1)
		shape(size_t w) { width = w; }

		//Constructor specifying width and height
		//Shape: (w, h, 1)
		shape(size_t w, size_t h) { width = w; height = h; }

		//Constructor specifying width, height and depth
		//Shape: (w, h, d)
		shape(size_t w, size_t h, size_t d) { width = w; height = h; depth = d; }

		//Size
		//Get the size of the shape (width * height * depth)
		const inline size_t size() { return width * height* depth; }

		//OPERATOR ==
		//Check if 2 shapes are equal in all dimensions
		const inline bool operator==(shape& B) { return (width == B.width && height == B.height && depth == B.depth); }

		//OPERATOR !=
		//Check if 2 shapes are inequal in any dimensions
		const inline bool operator!=(shape & B) { return (width != B.width || height != B.height || depth != B.depth); }

		//Str
		//Return a string representation of the shape in the format:
		//("width", "height", "depth")
		const inline string str() const { return "(" + to_string(width) + ", " + to_string(height) + ", " + to_string(depth) + ")"; }

		//API FUNCTION
		//Serialise
		//Returns a byte stream of the data stored by this shape
		const inline char* serialise() const {
			size_t shape_dims[3] = { width, height, depth };
			return reinterpret_cast<char*>(reinterpret_cast<void*>(shape_dims));
		}

		//API FUNCTION
		//Deserialise
		//Sets up this shape from a supplied byte stream
		inline void deserialise(char* stream_buffer, size_t offset) {
			size_t* shape_dims = reinterpret_cast<size_t*>(reinterpret_cast<void*>(&stream_buffer[offset]));
			width = shape_dims[0];
			height = shape_dims[1];
			depth = shape_dims[2];
		}

		//Max Dim
		//Returns the biggest dimension in this shape
		inline size_t max_dim() {
			size_t shape_dims[3] = { width, height, depth };
			return *max_element(&shape_dims[0], &shape_dims[2]);
		}
	};
}