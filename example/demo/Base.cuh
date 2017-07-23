#ifndef BASE_CUH
#define BASE_CUH

#include <curand_kernel.h>

#include "ParaCellsObject.cuh"

class Base
{
private:
	float *h_matrix;
	float *h_array;

	float *d_matrix;
	float *d_array;

	curandState *d_randStates;

	ParaCellsObject **d_paracellsObjects;

	int maxMatX;
	int currentMatX;

	int maxMatY;
	int currentMatY;

	int maxArrX;
	int currentArrX;

	//Flag
	bool hasUnpushedChangesInHost;
	bool hasUnpulledChangesInDevice;

public:
	Base(int maxMatX, int maxMatY, int maxArrX);
	virtual ~Base();

	int getMaxMatX();
	int getMaxMatY();
	int getMaxArrX();

	void setCurrentMatX(int value);
	int getCurrentMatX();

	void setCurrentMatY(int value);
	int getCurrentMatY();

	void setCurrentArrX(int value);
	int getCurrentArrX();

	//Memory copy
	void matrixCopyFromHostToDevice();
	void matrixCopyFromDeviceToHost();
	void arrayCopyFromHostToDevice();
	void arrayCopyFromDeviceToHost();

	//Synchronize changes between CPU and GPU
	void pushFromHostToDevice();
	void pullFromDeviceToHost();

	//Add fields operations
	void addMatColumns(int num, int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum);
	void addMatRow(int y, int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum);
	void addArrElement(int x, float value);
	
	//Update fields operations
	void updateMatAllColumns(int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum);
	void updateMatAllColumnsWithoutRefresh(int funcIndex, char **cellIdentifiers, int currentCellIdentifiersNum, char **envParamIdentifiers, int currentEnvParamIdentifiersNum);
	void updateMatElement(int x, int y, float value);
	void updateArrElement(int x, float value);

	//Get fields operations
	float getMatElement(int x, int y);
	float getArrElement(int x);

	//Calculate Statistical results
	float sumMatRow(int y);
};

#endif