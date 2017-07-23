#ifndef CUSTOMFUNCTIONS_CUH
#define CUSTOMFUNCTIONS_CUH

#include "CustomFunction.cuh"
#include <cstdio>

/*
**Step-1: define custom functions
*/

__device__ void InitCells(Cell *cell, Environment *env)
{
	cell->setAttribute("ancestor", cell->getCellId());
	cell->setAttribute("value", env->getUniformRandom());
}

__device__ void CellFateDicision(Cell *cell, Environment *env)
{
	if (cell->getAttribute("value") < env->getAttribute("threshold"))
	{
		cell->die();
	}
	else
	{
		cell->proliferate();
		float value = cell->getAttribute("value");
		cell->setFirstDaughterCellAttribute("value", value / 2.0f);
		cell->setSecondDaughterCellAttribute("value", value / 2.0f);
	}
}



/*
**Step-2: package callable custom functions
*/

__device__ CustomFunction customFunctions[2] = { InitCells, CellFateDicision };



/*
**Step-3: save this file as CustomFunctions.cuh
*/

#endif