#ifndef CELL_CUH
#define CELL_CUH

#include "cuda_runtime.h"

#include "ParaCellsObject.cuh"

class Cell
{
private:
	float *cellMatrix;
	int x;
	int maxX;
	int maxY;

	char **identifiers;
	int currentIdentifiersNum;

	float *daughter1Attributes;
	float *daughter2Attributes;
	int resultCellNum;

	ParaCellsObject *customObject;

	__device__ float *getCellMatrix();
	__device__ int getX();
	__device__ int getMaxX();
	__device__ int getMaxY();

	__device__ int findIdentifiers(const char *identifierName);

public:
	__device__ Cell(float *cellMatrix, int x, int maxX, int maxY, char **identifiers, int currentIdentifiersNum, ParaCellsObject *customObject);
	__device__ virtual ~Cell();
	
	__device__ int getCellId();
	__device__ void setAttribute(const char *attrName, float value);
	__device__ float getAttribute(const char *attrName);

	__device__ void setCustomObject(ParaCellsObject *pointer);
	__device__ ParaCellsObject *getCustomObject();

	__device__ void die();
	__device__ void proliferate();
	__device__ void setFirstDaughterCellAttribute(const char *attrName, float value);
	__device__ void setSecondDaughterCellAttribute(const char *attrName, float value);

	__device__ int getResultCellNum();
	__device__ float *getDaughter1Attributes();
	__device__ float *getDaughter2Attributes();

	__device__ void pushSelfToDaughter1();
};

#endif
