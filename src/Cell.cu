#include "Cell.cuh"

__device__ Cell::Cell(float *cellMatrix, int x, int maxX, int maxY, char **identifiers, int currentIdentifiersNum, ParaCellsObject *customObject)
{
	this->cellMatrix = cellMatrix;
	this->x = x;
	this->maxX = maxX;
	this->maxY = maxY;

	this->identifiers = identifiers;
	this->currentIdentifiersNum = currentIdentifiersNum;

	this->resultCellNum = 1;
	this->daughter1Attributes = 0;
	this->daughter2Attributes = 0;

	this->customObject = customObject;
}

__device__ float *Cell::getCellMatrix()
{
	return cellMatrix;
}

__device__ int Cell::getX()
{
	return x;
}

__device__ int Cell::getMaxX()
{
	return maxX;
}

__device__ int Cell::getMaxY()
{
	return maxY;
}

__device__ int Cell::findIdentifiers(const char *identifierName)
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

__device__ int Cell::getCellId()
{
	return getX();
}

__device__ void Cell::setAttribute(const char *attrName, float value)
{
	int index = findIdentifiers(attrName);
	if (index != -1)
	{
		cellMatrix[maxX*index + x] = value;
	}
}

__device__ float Cell::getAttribute(const char *attrName)
{
	int index = findIdentifiers(attrName);
	if (index != -1)
	{
		return cellMatrix[maxX*index + x];
	}

	return -1;
}

__device__ void Cell::setCustomObject(ParaCellsObject *pointer)
{
	if (customObject != 0)
	{
		delete customObject;
	}

	customObject = pointer;
}

__device__ ParaCellsObject *Cell::getCustomObject()
{
	return customObject;
}

__device__ void Cell::die()
{
	resultCellNum = 0;
}

__device__ void Cell::proliferate()
{
	if (resultCellNum != 1)
	{
		return;
	}

	resultCellNum = 2;
	daughter1Attributes = (float *)malloc(sizeof(float)*maxY);
	daughter2Attributes = (float *)malloc(sizeof(float)*maxY);

	for (int i = 0; i < maxY; i++)
	{
		daughter1Attributes[i] = cellMatrix[maxX*i + x];
		daughter2Attributes[i] = cellMatrix[maxX*i + x];
	}
}

__device__ void Cell::setFirstDaughterCellAttribute(const char *attrName, float value)
{
	if (resultCellNum != 2)
	{
		return;
	}

	int index = findIdentifiers(attrName);
	if (index != -1)
	{
		daughter1Attributes[index] = value;
	}
}

__device__ void Cell::setSecondDaughterCellAttribute(const char *attrName, float value)
{
	if (resultCellNum != 2)
	{
		return;
	}

	int index = findIdentifiers(attrName);
	if (index != -1)
	{
		daughter2Attributes[index] = value;
	}
}

__device__ int Cell::getResultCellNum()
{
	return resultCellNum;
}

__device__ float *Cell::getDaughter1Attributes()
{
	return daughter1Attributes;
}

__device__ float *Cell::getDaughter2Attributes()
{
	return daughter2Attributes;
}

__device__ void Cell::pushSelfToDaughter1()
{
	if (resultCellNum != 1)
	{
		return;
	}

	daughter1Attributes = (float *)malloc(sizeof(float)*maxY);

	for (int i = 0; i < maxY; i++)
	{
		daughter1Attributes[i] = cellMatrix[maxX*i + x];
	}
}

__device__ Cell::~Cell()
{
	free(daughter1Attributes);
	free(daughter2Attributes);
}