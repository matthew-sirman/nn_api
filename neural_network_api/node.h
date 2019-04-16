#pragma once

#include <vector>

using namespace std;

namespace nnet {
	namespace nnet_internal {
		template <typename T>
		struct node
		{
			node() {};

			node(T value) {
				this->value = value;
			}

			node(T value, vector<node<T>*> parents, vector<node<T>*> children) {
				this->value = value;
				this->parents = parents;
				this->children = children;
			}

			inline bool operator == (node<T> other) {
				return value == other.value &&
					parents == other.parents &&
					children == other.children &&
					activated == other.activated;
			}

			inline bool operator != (node<T> other) {
				return value != other.value ||
					parents != other.parents ||
					children != other.children ||
					activated != other.activated;
			}

			inline bool operator < (node<T> other) {
				return false;
			}

			T value;

			vector<node<T>*> parents;
			vector<node<T>*> children;

			bool activated = false;
		};
	}
}

