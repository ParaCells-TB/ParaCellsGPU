#include "BuiltInKernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CustomFunctions.cuh"

__global__ void addMatColumnsKernel(int currentMatX, int offset, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, curandState *randStates, ParaCellsObject **paracellsObjects)
{
	//Generate column index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int index = offset + idx;

	//Check validation of column index
	if (index >= currentMatX)
	{
		return;
	}

	//curandState localState = randStates[index];
	
	//Column elements initialization
	for (int i = 0; i < maxMatY; i++)
	{
		d_matrix[index + i*maxMatX] = 0;
	}

	if (funcIndex >= 0)
	{
		//Generate environment object
		Environment *env = new Environment(d_array, maxArrX, envParamIdentifiers, currentEnvParamIdentifiersNum, &randStates[index], currentMatX);

		//Generate cell object
		Cell *cell = new Cell(d_matrix, index, maxMatX, maxMatY, cellIdentifiers, currentCellIdentifiersNum, paracellsObjects[index]);

		//Call user-specified function
		customFunctions[funcIndex](cell, env);

		//Update custom objects pointer
		paracellsObjects[index] = cell->getCustomObject();

		//Free unused space
		delete env;
		delete cell;
	}

	//randStates[index] = localState;
}

__global__ void addMatRowKernel(int currentMatX, int y, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, curandState *randStates, ParaCellsObject **paracellsObjects)
{
	//Generate column index
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//Check validation of column index
	if (index >= currentMatX)
	{
		return;
	}

	//curandState localState = randStates[index];

	if (funcIndex >= 0)
	{
		//Generate environment object
		Environment *env = new Environment(d_array, maxArrX, envParamIdentifiers, currentEnvParamIdentifiersNum, &randStates[index], currentMatX);

		//Generate cell object
		Cell *cell = new Cell(d_matrix, index, maxMatX, maxMatY, cellIdentifiers, currentCellIdentifiersNum, paracellsObjects[index]);

		//Call user-specified function
		customFunctions[funcIndex](cell, env);

		//Update custom objects pointer
                paracellsObjects[index] = cell->getCustomObject();

		//Free unused space
		delete env;
		delete cell;
	}

	//randStates[index] = localState;
}

__global__ void updateMatColumnsKernel(int currentMatX, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, void **d_resultCells, int *d_resultCellNum, void **d_env, curandState *randStates, ParaCellsObject **paracellsObjects)
{
	//Generate column index
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//Check validation of column index
	if (index >= currentMatX)
	{
		return;
	}

	//curandState localState = randStates[index];

	if (funcIndex >= 0)
	{
		//Generate environment object
		Environment *env = new Environment(d_array, maxArrX, envParamIdentifiers, currentEnvParamIdentifiersNum, &randStates[index], currentMatX);

		//Generate cell object
		Cell *cell = new Cell(d_matrix, index, maxMatX, maxMatY, cellIdentifiers, currentCellIdentifiersNum, paracellsObjects[index]);

		//Call user-specified function
		customFunctions[funcIndex](cell, env);

		//Push cell attributes to daughter1 when not die or proliferate
		if (cell->getResultCellNum() == 1)
		{
			cell->pushSelfToDaughter1();
		}

		d_resultCells[index] = cell;
		d_resultCellNum[index] = cell->getResultCellNum();
		d_env[index] = env;

		//Free unused space
		//delete env;
		//delete cell;
	}

	//randStates[index] = localState;
}

__global__ void refreshMatColumnsByCellsKernel(int currentMatX, void **d_resultCells, int *d_resultCellNum, void **d_env, float *d_matrix, int maxMatX, int currentMatY, ParaCellsObject **paracellsObjects)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= currentMatX)
	{
		return;
	}

	//Get original cells with result
	Cell *cell = (Cell *)d_resultCells[index];

	//Get original environment
	Environment *env = (Environment *)d_env[index];

	//Check if cell has died
	if (cell->getResultCellNum() == 0)
	{
		if (cell->getCustomObject() != NULL)
		{
			delete (ParaCellsObject *)cell->getCustomObject();
		}
		delete cell;
		delete env;
		return;
	}

	//Calculate the cell id of new cells
	int new_index = d_resultCellNum[cell->getCellId()] - cell->getResultCellNum();
	
	//Get daughter cells' attribute
	float *daughter1 = cell->getDaughter1Attributes();
	float *daughter2 = cell->getDaughter2Attributes();

	//Overwrite the daughter cell to original matrix
	for (int i = 0; i < currentMatY; i++)
	{
		d_matrix[maxMatX*i + new_index] = daughter1[i];
		if (cell->getResultCellNum() == 2)
		{
			d_matrix[maxMatX*i + new_index + 1] = daughter2[i];
		}
	}

	//Overwrite the custom objects to original pointer array
	paracellsObjects[new_index] = cell->getCustomObject();
	if (cell->getResultCellNum() == 2)
	{
		if (paracellsObjects[new_index])
		{
			paracellsObjects[new_index + 1] = paracellsObjects[new_index]->proliferate(env);
		}
		else
		{
			paracellsObjects[new_index + 1] = 0;
		}
	}

	//Free memory space
	delete cell;
	delete env;
}

__global__ void addIdentifierKernel(char **identifiers, int index, char *identifierName)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx > 0)
	{
		return;
	}

	identifiers[index] = identifierName;
}

__global__ void freeIdentifierKernel(char **identifiers, int length)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx > 0)
	{
		return;
	}

	for (int i = 0; i < length; i++)
	{
		free(identifiers[i]);
	}
}

__global__ void curandSetupKernel(curandState *states, int size, int seed)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	curandState localState = states[idx];

	curand_init(seed, idx, 0, &localState);

	states[idx] = localState;
}

__global__ void paracellsObjectPointerSetupKernel(ParaCellsObject **paracellsObjects, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	paracellsObjects[idx] = 0;
}

__global__ void updateMatColumnsWithoutRefreshKernel(int currentMatX, int funcIndex, float *d_matrix, float* d_array, int maxMatX, int maxMatY, int maxArrX, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum, curandState *randStates, ParaCellsObject **paracellsObjects)
{
	//Generate column index
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//Check validation of column index
	if (index >= currentMatX)
	{
		return;
	}

	//curandState localState = randStates[index];

	if (funcIndex >= 0)
	{
		//Generate environment object
		Environment *env = new Environment(d_array, maxArrX, envParamIdentifiers, currentEnvParamIdentifiersNum, &randStates[index], currentMatX);

		//Generate cell object
		Cell *cell = new Cell(d_matrix, index, maxMatX, maxMatY, cellIdentifiers, currentCellIdentifiersNum, paracellsObjects[index]);

		//Call user-specified function
		customFunctions[funcIndex](cell, env);

		//Update custom objects pointer
		paracellsObjects[index] = cell->getCustomObject();

		//Free unused space
		delete env;
		delete cell;
	}

	//randStates[index] = localState;
}

__global__ void freeParaCellsObjectsKernel(ParaCellsObject **paracellsObjects, int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	if (paracellsObjects[idx] != NULL)
	{
		delete paracellsObjects[idx];
	}
}
