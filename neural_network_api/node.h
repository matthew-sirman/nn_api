#pragma once

#include <vector>

using namespace std;

namespace nnet {
	namespace nnet_internal {
		//Node
		//A single node present in a graph. Contains links to children and parents
		//(edges)
		//Takes a template argument for the graph type
		template <typename T>
		struct node
		{
			//Default constructor
			node() {};

			//Constructor specifying value
			node(T value) {
				this->value = value;
			}

			//Constructor specifying value, parents and children
			node(T value, vector<node<T>*> parents, vector<node<T>*> children) {
				this->value = value;
				this->parents = parents;
				this->children = children;
			}

			//Equality Operator
			inline bool operator == (node<T> other) {
				//returns true if all properties of nodes are equal (nodes are 
				//identical)
				return value == other.value &&
					parents == other.parents &&
					children == other.children &&
					activated == other.activated;
			}

			//Inequality Operator
			inline bool operator != (node<T> other) {
				//returns true if any properties are different
				return value != other.value ||
					parents != other.parents ||
					children != other.children ||
					activated != other.activated;
			}

			//Less Than Operator
			inline bool operator < (node<T> other) {
				//return false every time as there is no order
				//(needed for sets)
				return false;
			}

			//Value
			//The value of this node
			T value;

			//Parents
			//References to each parent to this node
			vector<node<T>*> parents;

			//Children
			//References to each child to this node
			vector<node<T>*> children;

			//Activated
			//Flag to determine if the node is actived (for graph traversal)
			bool activated = false;
		};
	}
}

