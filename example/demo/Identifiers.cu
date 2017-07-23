#include "Identifiers.cuh"

#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"

#include "BuiltInKernels.cuh"
#include "ParaCellsError.cuh"

Identifiers::Identifiers(int maxIdentifiersNum)
{
	//Size initialization
	this->maxIdentifiersNum = maxIdentifiersNum;
	this->currentIdentifiersNum = 0;

	//Unpushed data flag initialization
	this->hasUnpushedChangesInHost = 0;
	this->hasUnpulledChangesInDevice = 0;

	//CPU identifiers memory allocation
	h_identifiers = (char**)malloc(sizeof(char *)*maxIdentifiersNum);

	//GPU identifiers memory allocation
	CUDACall(cudaMalloc((void**)&d_identifiers, sizeof(char *)*maxIdentifiersNum));
}

char **Identifiers::getHostIdentifiers()
{
	return h_identifiers;
}

char **Identifiers::getDeviceIdentifiers()
{
	return d_identifiers;
}

int Identifiers::getMaxIdentifiersNum()
{
	return maxIdentifiersNum;
}

void Identifiers::setCurrentIdentifiersNum(int value)
{
	if (value > maxIdentifiersNum)
	{
		raiseError("The number of identifiers exceeded!");
	}

	currentIdentifiersNum = value;
}

int Identifiers::getCurrentIdentifiersNum()
{
	return currentIdentifiersNum;
}

void Identifiers::addIdentifier(int index, const char *identifierName)
{
	if (index >= maxIdentifiersNum)
	{
		raiseError("The index of identifier exceeded!");
	}

	setCurrentIdentifiersNum(currentIdentifiersNum + 1);

	int len = strlen(identifierName);

	//CPU memory allocation
	h_identifiers[index] = (char*)malloc(sizeof(char)*(len+1));

	//CPU string copy
	strcpy(h_identifiers[index], identifierName);

	//GPU string alloc&copy by kernel
	char *d_identifierName;
	CUDACall(cudaMalloc((void **)&d_identifierName, sizeof(char)*(len + 1)));
	CUDACall(cudaMemcpy(d_identifierName, identifierName, sizeof(char)*(len + 1), cudaMemcpyHostToDevice));
	addIdentifierKernel<<<1, 1>>>(d_identifiers, index, d_identifierName);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());
}

int Identifiers::findIdentifier(const char *identifierName)
{
	for (int i = 0; i < currentIdentifiersNum; i++)
	{
		if (strcmp(identifierName, h_identifiers[i]) == 0)
		{
			//Found
			return i;
		}
	}

	//Not found
	return -1;
}

Identifiers::~Identifiers()
{
	//free identifiers' memory on host
	for (int i = 0; i < currentIdentifiersNum; i++)
	{
		free(h_identifiers[i]);
	}
	free(h_identifiers);

	//free identifiers' memory on device
	freeIdentifierKernel<<<1, 1>>>(d_identifiers, currentIdentifiersNum);
	CUDACall(cudaGetLastError());
	CUDACall(cudaDeviceSynchronize());
	cudaFree(d_identifiers);
}