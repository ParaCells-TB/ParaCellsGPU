#ifndef CUSTOMFUNCTIONS_CUH
#define CUSTOMFUNCTIONS_CUH

#include "CustomFunction.cuh"

#include "TestObject.cuh"

/*
**Step-1: define custom functions
*/

__device__ void allCellsVal1PlusOne(Cell *cell, Environment *env)
{
	int value = cell->getAttribute("val1");
	cell->setAttribute("val1", value + 1);
}

__device__ void addAttributeVal2InitializedToCellId(Cell *cell, Environment *env)
{
	int cellId = cell->getCellId();
	cell->setAttribute("val2", cellId);
}

__device__ void addCellsInitialization(Cell *cell, Environment *env)
{
	int value = cell->getAttribute("val1");
	int cellId = cell->getCellId();
	cell->setAttribute("val1", value + 1);
	cell->setAttribute("val2", cellId);
}

__device__ void allCellsVal1ToEnv1AndVal2ToEnv2(Cell *cell, Environment *env)
{
	cell->setAttribute("val1", env->getAttribute("env1"));
	cell->setAttribute("val2", env->getAttribute("env2"));
}

__device__ void cellDeathAndProliferation(Cell *cell, Environment *env)
{
	if (cell->getCellId() % 2 == 0)
	{
		cell->die();
	}
	else if (cell->getCellId() % 3 == 0)
	{
		cell->proliferate();
	}
}

__device__ void generateRandomValue(Cell *cell, Environment *env)
{
	float val = env->getUniformRandom();
	cell->setAttribute("randomVal", val);
}

__device__ void generatePossionRandom(Cell *cell, Environment *env)
{
	float lambda = env->getAttribute("lambda");
	float val = env->getPossionRandom(lambda);
	cell->setAttribute("possionRandomVal", val);
}

__device__ void incrEnvironmentAttribute(Cell *cell, Environment *env)
{
	env->incrAttribute("attr", cell->getCellId());
}

__device__ void initCustomObject(Cell *cell, Environment *env)
{
	TestObject *obj = new TestObject(cell->getCellId());
	cell->setCustomObject(obj);
	cell->proliferate();
}

__device__ void obtainCustomObject(Cell *cell, Environment *env)
{
	TestObject *obj = (TestObject *)cell->getCustomObject();
	cell->setAttribute("val", obj->getVal());
}



/*
**Step-2: package callable custom functions
*/

__device__ CustomFunction customFunctions[10] = { allCellsVal1PlusOne, addAttributeVal2InitializedToCellId, addCellsInitialization, allCellsVal1ToEnv1AndVal2ToEnv2, cellDeathAndProliferation, generateRandomValue, generatePossionRandom, incrEnvironmentAttribute, initCustomObject, obtainCustomObject };



/*
**Step-3: save this file as CustomFunctions.cuh
*/

#endif
