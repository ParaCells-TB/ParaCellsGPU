#include "Base.cuh"

#include <cstdlib>
#include <cstdio>
#include <time.h>
#include "cuda_runtime.h"

#include "BuiltInKernels.cuh"
#include "ParaCellsError.cuh"

Base::Base(int maxMatX, int maxMatY, int maxArrX)
{
	//Maximum size initialization
	this->maxMatX = maxMatX;
	this->maxMatY = maxMatY;
	this->maxArrX = maxArrX;

	//Current size initialization
	this->currentMatX = 0;
	this->currentMatY = 0;
	this->currentArrX = 0;

	//Unpushed data flag initialization
	this->hasUnpushedChangesInHost = 0;
	this->hasUnpulledChangesInDevice = 0;
	
	//CPU cell matrix memory allocation
	h_matrix = (float *)malloc(sizeof(float)*maxMatY*maxMatX);

	//CPU environment parameters memory allocation
	h_array = (float *)malloc(sizeof(float)*maxArrX);

	//GPU cell matrix memory allocation
	CUDACall(cudaMalloc((void**)&d_matrix, sizeof(float)*maxMatY*maxMatX));

	//GPU environment parameters memory allocation
	CUDACall(cudaMalloc((void**)&d_array, sizeof(float)*maxArrX));

	//GPU curand states memory allocation
	CUDACall(cudaMalloc((void**)&d_randStates, sizeof(curandState)*maxMatX));

	//GPU ParaCellsObject pointers memory allocation
	CUDACall(cudaMalloc((void**)&d_paracellsObjects, sizeof(ParaCellsObject *)*maxMatX));

	//GPU curand states initialization in kernel
	curandSetupKernel<<<maxMatX/256+1, 256>>>(d_randStates, maxMatX, time(NULL));
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());

	//GPU ParaCellsObject pointers initialization in kernel
	paracellsObjectPointerSetupKernel<<<maxMatX/256+1, 256>>>(d_paracellsObjects, maxMatX);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());
}

int Base::getMaxMatX()
{
	return maxMatX;
}

int Base::getMaxMatY()
{
	return maxMatY;
}

int Base::getMaxArrX()
{
	return maxArrX;
}

void Base::setCurrentMatX(int value)
{
	if (value > maxMatX)
	{
		raiseError("The number of cells exceeded!");
	}
	currentMatX = value;
}

int Base::getCurrentMatX()
{
	return currentMatX;
}

void Base::setCurrentMatY(int value)
{
	if (value > maxMatY)
	{
		raiseError("The number of cell attributes exceeded!");
	}
	currentMatY = value;
}

int Base::getCurrentMatY()
{
	return currentMatY;
}

void Base::setCurrentArrX(int value)
{
	if (value > maxArrX)
	{
		raiseError("The number of environment parameters exceeded!");
	}
	currentArrX = value;
}

int Base::getCurrentArrX()
{
	return currentArrX;
}

void Base::matrixCopyFromHostToDevice()
{
	//Copy cell matrix from CPU to GPU
	CUDACall(cudaMemcpy(d_matrix, h_matrix, sizeof(float)*maxMatY*maxMatX, cudaMemcpyHostToDevice));
}

void Base::matrixCopyFromDeviceToHost()
{
	//Copy cell matrix from GPU to CPU
	CUDACall(cudaMemcpy(h_matrix, d_matrix, sizeof(float)*maxMatY*maxMatX, cudaMemcpyDeviceToHost));
}

void Base::arrayCopyFromHostToDevice()
{
	//Copy environment parameters from CPU to GPU
	CUDACall(cudaMemcpy(d_array, h_array, sizeof(float)*maxArrX, cudaMemcpyHostToDevice));
}

void Base::arrayCopyFromDeviceToHost()
{
	//Copy environment parameters from GPU to CPU
	CUDACall(cudaMemcpy(h_array, d_array, sizeof(float)*maxArrX, cudaMemcpyDeviceToHost));
}

void Base::pushFromHostToDevice()
{
	if (hasUnpushedChangesInHost)
	{
		hasUnpushedChangesInHost = 0;
		matrixCopyFromHostToDevice();
		arrayCopyFromHostToDevice();
	}
}

void Base::pullFromDeviceToHost()
{
	if (hasUnpulledChangesInDevice)
	{
		hasUnpulledChangesInDevice = 0;
		matrixCopyFromDeviceToHost();
		arrayCopyFromDeviceToHost();
	}
}

void Base::addMatColumns(int num, int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum)
{
	//Update cell size
	setCurrentMatX(currentMatX + num);

	//Push changes from CPU to GPU
	pushFromHostToDevice();

	//Added cells initialization (using user-specified function)
	addMatColumnsKernel<<<num/256+1, 256>>>(currentMatX, currentMatX - num, funcIndex, d_matrix, d_array, maxMatX, maxMatY, maxArrX, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum, d_randStates, d_paracellsObjects);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());

	hasUnpulledChangesInDevice = 1;
}

void Base::addMatRow(int y, int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum)
{
	if (y >= maxMatY)
	{
		raiseError("The index of cell attribute exceeded!");
	}

	//Update attribute size
	setCurrentMatY(currentMatY + 1);

	//Push changes from CPU to GPU
	pushFromHostToDevice();

	//All cells update (using user-specified function), in order to initialize newly added attribute
	addMatRowKernel<<<currentMatX/256+1, 256>>>(currentMatX, y, funcIndex, d_matrix, d_array, maxMatX, maxMatY, maxArrX, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum, d_randStates, d_paracellsObjects);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());

	hasUnpulledChangesInDevice = 1;
}

void Base::addArrElement(int x, float value)
{
	if (x >= maxArrX)
	{
		raiseError("The index of environment parameter exceeded!");
	}

	//Update environment parameter size
	setCurrentArrX(currentArrX + 1);

	//Pull changes from GPU to CPU
	pullFromDeviceToHost();

	//Add environment parameter
	h_array[x] = value;

	hasUnpushedChangesInHost = 1;
}

void Base::updateMatAllColumns(int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum)
{
	//Push changes from CPU to GPU
	pushFromHostToDevice();

	//Cell update results
	int *h_resultCellNum = (int *)malloc(sizeof(int)*currentMatX);
	int *d_resultCellNum;
	CUDACall(cudaMalloc((void**)&d_resultCellNum, sizeof(int)*currentMatX));
	void **d_resultCells;
	CUDACall(cudaMalloc((void**)&d_resultCells, sizeof(size_t)*currentMatX));

	//Keeping environment objects
	void **d_env;
	CUDACall(cudaMalloc((void**)&d_env, sizeof(size_t)*currentMatX));

	//All cells update (using user-specified function)
	updateMatColumnsKernel<<<currentMatX/256+1, 256>>>(currentMatX, funcIndex, d_matrix, d_array, maxMatX, maxMatY, maxArrX, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum, d_resultCells, d_resultCellNum, d_env, d_randStates, d_paracellsObjects);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());

	//Copy result cell num from GPU to CPU
	CUDACall(cudaMemcpy(h_resultCellNum, d_resultCellNum, sizeof(int)*currentMatX, cudaMemcpyDeviceToHost));

	//Reduce in serial computing
	for (int i = 1; i < currentMatX; i++)
	{
		h_resultCellNum[i] += h_resultCellNum[i - 1];
	}

	//Copy result cell num from CPU to GPU
	CUDACall(cudaMemcpy(d_resultCellNum, h_resultCellNum, sizeof(int)*currentMatX, cudaMemcpyHostToDevice));

	//Check result cell num is valid
	if (h_resultCellNum[currentMatX - 1] > maxMatX)
	{
		raiseError("The number of cells exceeded!");
	}

	//Dispatch new cells info back to matrix
	refreshMatColumnsByCellsKernel <<<currentMatX/256+1, 256>>>(currentMatX, d_resultCells, d_resultCellNum, d_env, d_matrix, maxMatX, currentMatY, d_paracellsObjects);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());

	//Update current matrix dimention
	setCurrentMatX(h_resultCellNum[currentMatX - 1]);

	//Free memory space
	free(h_resultCellNum);
	cudaFree(d_resultCellNum);
	cudaFree(d_resultCells);
	cudaFree(d_env);

	hasUnpulledChangesInDevice = 1;
}

void Base::updateMatAllColumnsWithoutRefresh(int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum)
{
	//Push changes from CPU to GPU
	pushFromHostToDevice();

	//All cells update without refresh (using user-specified function)
	updateMatColumnsWithoutRefreshKernel<<<currentMatX/256+1, 256>>>(currentMatX, funcIndex, d_matrix, d_array, maxMatX, maxMatY, maxArrX, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum, d_randStates, d_paracellsObjects);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());

	hasUnpulledChangesInDevice = 1;
}

void Base::updateMatElement(int x, int y, float value)
{
	if (x >= maxMatX)
	{
		raiseError("The index of cell exceeded!");
	}

	if (y >= maxMatY)
	{
		raiseError("The index of cell attribute exceeded!");
	}

	//Pull changes from GPU to CPU
	pullFromDeviceToHost();

	//Update matrix element
	h_matrix[x + y*maxMatX] = value;

	hasUnpushedChangesInHost = 1;
}

void Base::updateArrElement(int x, float value)
{
	if (x >= maxArrX)
	{
		raiseError("The index of environment parameters exceeded!");
	}

	//Pull changes from GPU to CPU
	pullFromDeviceToHost();

	//Update environment parameter
	h_array[x] = value;

	hasUnpushedChangesInHost = 1;
}

float Base::getMatElement(int x, int y)
{
	if (x >= maxMatX)
	{
		raiseError("The index of cell exceeded!");
	}

	if (y >= maxMatY)
	{
		raiseError("The index of cell attribute exceeded!");
	}

	//Pull changes from GPU to CPU
	pullFromDeviceToHost();

	return h_matrix[x + y*maxMatX];
}

float Base::getArrElement(int x)
{
	if (x >= maxArrX)
	{
		raiseError("The index of environment parameter exceeded!");
	}

	//Pull changes from GPU to CPU
	pullFromDeviceToHost();

	return h_array[x];
}

float Base::sumMatRow(int y)
{
	//Pull changes from GPU to CPU
	pullFromDeviceToHost();

	float rtn = 0;
	int offset = y*maxMatX;

	for (int i = 0; i < currentMatX; i++)
	{
		rtn += h_matrix[offset + i];
	}

	return rtn;
}

Base::~Base()
{
	free(h_matrix);
	free(h_array);

	freeParaCellsObjectsKernel<<<currentMatX/256+1, 256>>>(d_paracellsObjects, currentMatX);

	cudaFree(d_matrix);
	cudaFree(d_array);
	cudaFree(d_randStates);
	cudaFree(d_paracellsObjects);
}
