#include <iostream>
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>

#include "CellEnv.cuh"

#define	CELL_NUM 10000000

using namespace std;

void work()
{
	CellEnv cellEnv(CELL_NUM, 2, 0);
	cellEnv.addCells(CELL_NUM);
	cellEnv.addCellAttribute("P0");
	cellEnv.addCellAttribute("p53");
	cellEnv.updateAllCellsWithoutFateDecision(0);
	cellEnv.updateAllCellsWithoutFateDecision(1);
}

int main()
{
	struct timeval start, end;
	int milliseconds;

	gettimeofday(&start, NULL);
	
	work();
	
	gettimeofday(&end, NULL);

	milliseconds = ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000.0) + 0.5;

	printf("Time Calc: %.2fs\n", (double)milliseconds/1000.0);
	return 0;
}
