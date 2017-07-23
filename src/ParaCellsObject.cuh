#ifndef PARACELLSOBJECT_CUH
#define PARACELLSOBJECT_CUH

#include "Environment.cuh"

class ParaCellsObject
{
public:
	__device__ virtual ParaCellsObject *proliferate(Environment *env)=0;

	__device__ virtual ~ParaCellsObject()
	{
		//Nothing
	}
};

#endif
