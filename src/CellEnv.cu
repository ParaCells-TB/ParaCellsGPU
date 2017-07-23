#include "CellEnv.cuh"

#include <cstdio>

CellEnv::CellEnv(int maxCellNum, int maxAttributeNum, int maxEnvironmentParameterNum)
{
	//Prevent overflow errors
	if (maxAttributeNum == 0) maxAttributeNum = 1;
	if (maxEnvironmentParameterNum == 0) maxEnvironmentParameterNum = 1;	

	base = new Base(maxCellNum, maxAttributeNum, maxEnvironmentParameterNum);

	cellAttributeIdentifiers = new Identifiers(maxAttributeNum);
	environmentParametersIdentifiers = new Identifiers(maxEnvironmentParameterNum);
}

int CellEnv::getMaxCellNum()
{
	return base->getMaxMatX();
}

int CellEnv::getMaxCellAttributeNum()
{
	return base->getMaxMatY();
}

int CellEnv::getMaxEnvironmentAttributeNum()
{
	return base->getMaxArrX();
}

int CellEnv::getCurrentCellNum()
{
	return base->getCurrentMatX();
}

int CellEnv::getCurrentCellAttributeNum()
{
	return base->getCurrentMatY();
}

int CellEnv::getCurrentEnvironmentAttributeNum()
{
	return base->getCurrentArrX();
}

void CellEnv::addCells(int num, int funcIndex)
{
	//Obtain identifiers information (for kernel)
	char **cellIdentifiers = cellAttributeIdentifiers->getDeviceIdentifiers();
	char **envParamIdentifiers = environmentParametersIdentifiers->getDeviceIdentifiers();
	int currentCellIdentifiersNum = cellAttributeIdentifiers->getCurrentIdentifiersNum();
	int currentEnvParamIdentifiersNum = environmentParametersIdentifiers->getCurrentIdentifiersNum();

	base->addMatColumns(num, funcIndex, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum);
}

void CellEnv::addCellAttribute(const char *attrName, int funcIndex)
{
	int index = getCurrentCellAttributeNum();
	
	//Add identifiers
	cellAttributeIdentifiers->addIdentifier(index, attrName);

	//Obtain identifiers information (for kernel)
	char **cellIdentifiers = cellAttributeIdentifiers->getDeviceIdentifiers();
	char **envParamIdentifiers = environmentParametersIdentifiers->getDeviceIdentifiers();
	int currentCellIdentifiersNum = cellAttributeIdentifiers->getCurrentIdentifiersNum();
	int currentEnvParamIdentifiersNum = environmentParametersIdentifiers->getCurrentIdentifiersNum();

	//Add attribute and initialization
	base->addMatRow(index, funcIndex, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum);
}

void CellEnv::addEnvironmentAttribute(const char *paramName, float value)
{
	int index = getCurrentEnvironmentAttributeNum();

	//Add identifiers
	environmentParametersIdentifiers->addIdentifier(index, paramName);

	//Add environment parameter
	base->addArrElement(index, value);
}

void CellEnv::updateAllCells(int funcIndex)
{
	//Obtain identifiers information (for kernel)
	char **cellIdentifiers = cellAttributeIdentifiers->getDeviceIdentifiers();
	char **envParamIdentifiers = environmentParametersIdentifiers->getDeviceIdentifiers();
	int currentCellIdentifiersNum = cellAttributeIdentifiers->getCurrentIdentifiersNum();
	int currentEnvParamIdentifiersNum = environmentParametersIdentifiers->getCurrentIdentifiersNum();

	base->updateMatAllColumns(funcIndex, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum);
}

void CellEnv::updateAllCellsWithoutFateDecision(int funcIndex)
{
	//Obtain identifiers information (for kernel)
	char **cellIdentifiers = cellAttributeIdentifiers->getDeviceIdentifiers();
	char **envParamIdentifiers = environmentParametersIdentifiers->getDeviceIdentifiers();
	int currentCellIdentifiersNum = cellAttributeIdentifiers->getCurrentIdentifiersNum();
	int currentEnvParamIdentifiersNum = environmentParametersIdentifiers->getCurrentIdentifiersNum();

	base->updateMatAllColumnsWithoutRefresh(funcIndex, cellIdentifiers, currentCellIdentifiersNum, envParamIdentifiers, currentEnvParamIdentifiersNum);
}

void CellEnv::setCellAttribute(int cellId, const char *attrName, float value)
{
	int index = cellAttributeIdentifiers->findIdentifier(attrName);
	if (index != -1)
	{
		//Update cell attribute
		base->updateMatElement(cellId, index, value);
	}
}

void CellEnv::setEnvironmentAttribute(const char *paramName, float value)
{
	int index = environmentParametersIdentifiers->findIdentifier(paramName);
	if (index != -1)
	{
		//Update environment parameter
		base->updateArrElement(index, value);
	}
}

float CellEnv::getCellAttribute(int cellId, const char *attrName)
{
	int index = cellAttributeIdentifiers->findIdentifier(attrName);
	if (index != -1)
	{
		return base->getMatElement(cellId, index);
	}
	return -1;
}

float CellEnv::getEnvironmentAttribute(const char *paramName)
{
	int index = environmentParametersIdentifiers->findIdentifier(paramName);
	if (index != -1)
	{
		return base->getArrElement(index);
	}
	return -1;
}

float CellEnv::sumAttributeFromAllCells(const char *attrName)
{
	int index = cellAttributeIdentifiers->findIdentifier(attrName);
	if (index != -1)
	{
		return base->sumMatRow(index);
	}
	return -1;
}

void CellEnv::printCells()
{
	char **identifiers = cellAttributeIdentifiers->getHostIdentifiers();
	int currentIdentifiersNum = cellAttributeIdentifiers->getCurrentIdentifiersNum();
	printf("==========Cells (%d)==========\n", base->getCurrentMatX());
	printf("Cell-ID");
	for (int i = 0; i < currentIdentifiersNum; i++)
	{
		printf("\t%s", identifiers[i]);
	}
	printf("\n");
	for (int i = 0; i < base->getCurrentMatX(); i++)
	{
		printf("%d", i);
		for (int j = 0; j < currentIdentifiersNum; j++)
		{
			printf("\t%f", base->getMatElement(i, cellAttributeIdentifiers->findIdentifier(identifiers[j])));
		}
		printf("\n");
	}
	printf("\n");
}

void CellEnv::printEnvironmentAttributes()
{
	char **identifiers = environmentParametersIdentifiers->getHostIdentifiers();
	int currentIdentifiersNum = environmentParametersIdentifiers->getCurrentIdentifiersNum();
	printf("==========Environment Attributes (%d)==========\n", currentIdentifiersNum);
	printf("Name\tValue\n");
	for (int i = 0; i < currentIdentifiersNum; i++)
	{
		printf("%s\t%f\n", identifiers[i], base->getArrElement(environmentParametersIdentifiers->findIdentifier(identifiers[i])));
	}
	printf("\n");
}

void CellEnv::print()
{
	printEnvironmentAttributes();
	printCells();
}

CellEnv::~CellEnv()
{
	delete base;
	delete cellAttributeIdentifiers;
	delete environmentParametersIdentifiers;
}
