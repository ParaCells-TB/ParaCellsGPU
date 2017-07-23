#ifndef CELLENV_CUH
#define CELLENV_CUH

#include "Base.cuh"
#include "Identifiers.cuh"

class CellEnv
{
private:
	Base *base;

	Identifiers *cellAttributeIdentifiers;
	Identifiers *environmentParametersIdentifiers;

public:
	CellEnv(int maxCellNum, int maxAttributeNum, int maxEnvironmentParameterNum);
	virtual ~CellEnv();

	int getMaxCellNum();
	int getMaxCellAttributeNum();
	int getMaxEnvironmentAttributeNum();

	int getCurrentCellNum();
	int getCurrentCellAttributeNum();
	int getCurrentEnvironmentAttributeNum();

	//Add operations
	void addCells(int num, int funcIndex = -1);
	void addCellAttribute(const char *attrName, int funcIndex = -1);
	void addEnvironmentAttribute(const char *paramName, float value = 0);

	//Set operations
	void updateAllCells(int funcIndex = -1);
	void updateAllCellsWithoutFateDecision(int funcIndex = -1);
	void setCellAttribute(int cellId, const char *attrName, float value);
	void setEnvironmentAttribute(const char *paramName, float value);

	//Get operations
	//Cell *getCellById(int cellId);
	//Environment *getEnvironment();
	float getCellAttribute(int cellId, const char *attrName);
	float getEnvironmentAttribute(const char *paramName);

	//Statistical operations
	float sumAttributeFromAllCells(const char *attrName);

	//Print cell environment
	void printCells();
	void printEnvironmentAttributes();
	void print();
};

#endif
