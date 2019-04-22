#pragma once

#include <string>

using namespace std;

namespace nnet {
	//Placeholder
	//A placeholder variable for feeding information into a function
	class placeholder {
	public:
		//Default constructor for an unnamed placeholder
		placeholder() {
			this->name = "unnamed";
		}

		//Constructor specifying placeholder name
		placeholder(string name) {
			this->name = name;
		}

		//Name
		//The name of this placeholder (default: "unnamed")
		string name;
	};
}