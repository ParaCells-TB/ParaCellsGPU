#ifndef CUSTOMFUNCTIONS_CUH
#define CUSTOMFUNCTIONS_CUH

#include "CustomFunction.cuh"

#include "C.cuh"

/*
**Step-1: define custom functions
*/

__device__ void init(Cell *cell, Environment *env)
{
	C *obj = new C(env);
	cell->setCustomObject(obj);
	cell->setAttribute("accumulator", 0);
	cell->setAttribute("pos_x", env->getUniformRandom()*60);
	cell->setAttribute("pos_y", env->getUniformRandom()*60);
	cell->setAttribute("orig_id", cell->getCellId());
}

__device__ void updatePosition(Cell *cell, Environment *env)
{
	float x = cell->getAttribute("pos_x");
	float y = cell->getAttribute("pos_y");
	
	cell->setAttribute("last_pos_x", x);
	cell->setAttribute("last_pos_y", y);

	x += 2 * (env->getUniformRandom() - 0.5) * 3;
	y += 2 * (env->getUniformRandom() - 0.5) * 3;
	cell->setAttribute("pos_x", x);
	cell->setAttribute("pos_y", y);
}

__device__ void oneStep(Cell *cell, Environment *env)
{
	cell->setAttribute("accumulator", 0);
	C *obj = (C *)cell->getCustomObject();
	//if (!obj->alive()) return;

	for (int i = 0; i < STEP_NUM/10; i++)
	{
		cell->setAttribute("accumulator", 0);
		obj->oneStep(cell, env);
		if (!obj->alive()) break;
	}

	env->incrAttribute("accumulator", cell->getAttribute("accumulator"));
	updatePosition(cell, env);
}

__device__ void oneStepWithProliferation(Cell *cell, Environment *env)
{
	cell->setAttribute("accumulator", 0);
	C *obj = (C *)cell->getCustomObject();

	if (obj->alive())
	{
		obj->oneStep(cell, env);
		if (obj->isDivided(env))
		{
			cell->proliferate();
		}
	}
	else
	{
		cell->die();
	}

	env->incrAttribute("accumulator", cell->getAttribute("accumulator"));
	updatePosition(cell, env);
}



/*
**Step-2: package callable custom functions
*/

__device__ CustomFunction customFunctions[3] = { init, oneStep, oneStepWithProliferation };



/*
**Step-3: save this file as CustomFunctions.cuh
*/

#endif
