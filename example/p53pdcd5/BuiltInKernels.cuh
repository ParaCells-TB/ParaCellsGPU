#ifndef BUILTINKERNELS_CUH
#define BUILTINKERNELS_CUH

#include "cuda_runtime.h"
#include <curand_kernel.h>

#include "ParaCellsObject.cuh"

__global__ void addMatColumnsKernel(int currentMatX, int offset, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, curandState *randStates, ParaCellsObject **paracellsObjects);

__global__ void addMatRowKernel(int currentMatX, int y, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, curandState *randStates, ParaCellsObject **paracellsObjects);

__global__ void updateMatColumnsKernel(int currentMatX, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, void **d_resultCells, int *d_resultCellNum, void **d_env, curandState *randStates, ParaCellsObject **paracellsObjects);

__global__ void refreshMatColumnsByCellsKernel(int currentMatX, void **d_resultCells, int *d_resultCellNum, void **d_env, float *d_matrix, int maxMatX, int currentMatY, ParaCellsObject **paracellsObjects);

__global__ void addIdentifierKernel(char **identifiers, int index, char *identifierName);

__global__ void freeIdentifierKernel(char **identifiers, int length);

__global__ void curandSetupKernel(curandState *states, int size, int seed);

__global__ void paracellsObjectPointerSetupKernel(ParaCellsObject **paracellsObjects, int size);

__global__ void updateMatColumnsWithoutRefreshKernel(int currentMatX, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, curandState *randStates, ParaCellsObject **paracellsObjects);

__global__ void freeParaCellsObjectsKernel(ParaCellsObject **paracellsObjects, int size);

#endif
