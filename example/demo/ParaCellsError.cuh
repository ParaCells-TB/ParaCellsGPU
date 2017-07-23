#ifndef PARACELLSERROR_CUH
#define PARACELLSERROR_CUH

#include "cuda_runtime.h"

class ParaCellsError
{
private:
	char *message;

public:
	ParaCellsError();
	ParaCellsError(const char *message, const char *file, int line, int isFromCUDA);
	virtual ~ParaCellsError();

	char *getMessage();
};

#define raiseError(msg) { throw ParaCellsError((msg), __FILE__, __LINE__, 0); }

#define CUDACall(status) { _CUDACall((status), __FILE__, __LINE__); }
inline void _CUDACall(cudaError_t status, const char *file, int line)
{
	if (status != cudaSuccess)
	{
		throw ParaCellsError(cudaGetErrorString(status), file, line, 1);
	}
}

#endif