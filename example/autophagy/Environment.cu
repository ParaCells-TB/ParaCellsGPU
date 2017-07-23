#include "Environment.cuh"

__device__ Environment::Environment(float *envArray, int maxX, char **identifiers, int currentIdentifiersNum, curandState *randState, int currentCellNum)
{
	this->envArray = envArray;
	this->maxX = maxX;

	this->identifiers = identifiers;
	this->currentIdentifiersNum = currentIdentifiersNum;

	this->randState = randState;

	this->currentCellNum = currentCellNum;
}

__device__ float *Environment::getEnvArray()
{
	return envArray;
}

__device__ int Environment::getMaxX()
{
	return maxX;
}

__device__ int Environment::findIdentifiers(const char *identifierName)
{
	int j;

	for (int i = 0; i < currentIdentifiersNum; i++)
	{
		j = 0;
		while (identifierName[j] != '\0' && identifiers[i][j] != '\0')
		{
			if (identifierName[j] != identifiers[i][j])
			{
				break;
			}
			j++;
		}
		if (identifierName[j] == '\0' && identifiers[i][j] == '\0')
		{
			//Found
			return i;
		}
	}

	//Not found
	return -1;
}

//__device__ void Environment::setAttribute(const char *paramName, float value)
//{
//	int index = findIdentifiers(attrName);
//	if (index != -1)
//	{
//		envArray[index] = value;
//	}
//}

__device__ float Environment::getAttribute(const char *paramName)
{
	int index = findIdentifiers(paramName);
	if (index != -1)
	{
		return envArray[index];
	}

	return -1;
}

__device__ void Environment::incrAttribute(const char *paramName, float value)
{
	int index = findIdentifiers(paramName);
	if (index != -1)
	{
		atomicAdd(envArray + index, value);
	}
}

__device__ float Environment::getUniformRandom()
{
	return curand_uniform(randState);
}

__device__ float Environment::getPossionRandom(float lambda)
{
	return curand_poisson(randState, lambda);

	//Legacy implementation
	/*float x = -1, u;
	float log1, log2;
	log1 = 0;
	log2 = -lambda;
	do
	{
		u = getUniformRandom();
		log1 += log(u);
		x++;
	} while (log1 >= log2);
	return x;*/
}

__device__ int Environment::getCurrentCellNum()
{
	return currentCellNum;
}
