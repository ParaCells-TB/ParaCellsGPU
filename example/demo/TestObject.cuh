#ifndef TESTOBJECT_CUH
#define TESTOBJECT_CUH

#include "ParaCellsObject.cuh"

class TestObject: public ParaCellsObject
{
private:
	float val;

public:
	__device__ TestObject(float value)
	{
		val = value;
	}

	__device__ ParaCellsObject *proliferate(Environment *env)
	{
		TestObject *rtn = new TestObject(val + 1);
		return rtn;
	}

	__device__ float getVal()
	{
		return val;
	}
};

#endif
