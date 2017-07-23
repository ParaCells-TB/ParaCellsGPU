#ifndef CUSTOMFUNCTION_CUH
#define CUSTOMFUNCTION_CUH

#include "cuda_runtime.h"

#include "Cell.cuh"
#include "Environment.cuh"

typedef void(*CustomFunction)(Cell *, Environment *);

#endif