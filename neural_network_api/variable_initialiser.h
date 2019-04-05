#pragma once
struct variable_initialiser
{
public:
	variable_initialiser(float mean = 0.0f, float stddev = 0.01f);
	~variable_initialiser();

	float mean;
	float stddev;
};

