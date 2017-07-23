#ifndef ENVIRONMENT_CUH
#define ENVIRONMENT_CUH

#include "cuda_runtime.h"
#include <curand_kernel.h>

class Environment
{
private:
	float *envArray;
	int maxX;

	char **identifiers;
	int currentIdentifiersNum;

	curandState *randState;

	int currentCellNum;

	__device__ float *getEnvArray();
	__device__ int getMaxX();

	__device__ int findIdentifiers(const char *identifierName);

public:
	__device__ Environment(float *envArray, int maxX, char **identifiers, int currentIdentifiersNum, curandState *randState, int currentCellNum);

	//__device__ void setAttribute(const char *paramName, float value);
	__device__ float getAttribute(const char *paramName);
	__device__ void incrAttribute(const char *paramName, float value);

	__device__ float getUniformRandom();
	__device__ float getPossionRandom(float lambda);

	__device__ int getCurrentCellNum();
};

#endif
