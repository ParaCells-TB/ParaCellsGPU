#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>

#include "CellEnv.cuh"
#include "ParaCellsError.cuh"
#include "Constants.h"

using namespace std;

ofstream out23("aa-23.txt");
ofstream out24("aa-24.txt");
ofstream out25("aa-25.txt");
ofstream out26("aa-26.txt");
ofstream out27("aa-27.txt");

ofstream outPos("pos.txt");
int step_count = 0;

float alpha_t;

void updateAaout(CellEnv &cellEnv)
{
	float val = v * alpha_t * cellEnv.getCurrentCellNum();
	val += cellEnv.getEnvironmentAttribute("accumulator");
	val *= STEP_TIME;
	val += cellEnv.getEnvironmentAttribute("aaout");
	cellEnv.setEnvironmentAttribute("aaout", val);
	cellEnv.setEnvironmentAttribute("accumulator", 0);
}

void work()
{
	CellEnv cellEnv(MAXIMUM_CELL_NUM, 7, 2);

	cellEnv.addCellAttribute("orig_id");
	cellEnv.addCellAttribute("last_pos_x");
	cellEnv.addCellAttribute("last_pos_y");
	cellEnv.addCellAttribute("pos_x");
	cellEnv.addCellAttribute("pos_y");
	cellEnv.addCellAttribute("aa");
	cellEnv.addCellAttribute("accumulator");
	cellEnv.addEnvironmentAttribute("aaout", 0);
	cellEnv.addEnvironmentAttribute("accumulator", 0);

	cellEnv.addCells(INITIAL_CELL_NUM, 0);

	for (int i = 0; i < CYCLE_NUM; i++)
	{
		cout << i << "\t" << cellEnv.getCurrentCellNum() << endl;
		if (i % 3 == 0)
		{
			if ((i / 3) % 2 == 0) alpha_t = 0.125;
			else alpha_t = 0;
		}
		for (int j = 0; j <= 10; j++)
		{
			cellEnv.updateAllCellsWithoutFateDecision(1);
			updateAaout(cellEnv);

			step_count++;
			for (int k = 0; k < cellEnv.getCurrentCellNum(); k++)
				if (abs(cellEnv.getCellAttribute(k, "orig_id") - 150) < 1e-5)
				{
					outPos << step_count << "\t"
					<< cellEnv.getCellAttribute(k, "last_pos_x") << "\t"
					<< cellEnv.getCellAttribute(k, "last_pos_y") << "\t"
					<< cellEnv.getCellAttribute(k, "pos_x") << "\t"
					<< cellEnv.getCellAttribute(k, "pos_y") << endl;
				}
		}
		
		cellEnv.updateAllCells(2);
		updateAaout(cellEnv);

		step_count++;
                for (int k = 0; k < cellEnv.getCurrentCellNum(); k++)
                        if (abs(cellEnv.getCellAttribute(k, "orig_id") - 150) < 1e-5)
                        {
                                outPos << step_count << "\t" 
                                << cellEnv.getCellAttribute(k, "last_pos_x") << "\t"
                                << cellEnv.getCellAttribute(k, "last_pos_y") << "\t"
                                << cellEnv.getCellAttribute(k, "pos_x") << "\t"
                                << cellEnv.getCellAttribute(k, "pos_y") << endl;
                        }
		
		if (i == 23)
		{
			for (int j = 0; j <= cellEnv.getCurrentCellNum(); j++)
				out23 << fixed << setprecision(10) << j << "\t" << cellEnv.getCellAttribute(j, "aa")<<endl;
		}
		if (i == 24)
                {
                        for (int j = 0; j <= cellEnv.getCurrentCellNum(); j++)
                                out24 << fixed << setprecision(10) << j << "\t" << cellEnv.getCellAttribute(j, "aa")<<endl;
                }
		if (i == 25)
                {
                        for (int j = 0; j <= cellEnv.getCurrentCellNum(); j++)
                                out25 << fixed << setprecision(10) << j << "\t" << cellEnv.getCellAttribute(j, "aa")<<endl;
                }
		if (i == 26)
                {
                        for (int j = 0; j <= cellEnv.getCurrentCellNum(); j++)
                                out26 << fixed << setprecision(10) << j << "\t" << cellEnv.getCellAttribute(j, "aa")<<endl;
                }
		if (i == 27)
                {
                        for (int j = 0; j <= cellEnv.getCurrentCellNum(); j++)
                                out27 << fixed << setprecision(10) << j << "\t" << cellEnv.getCellAttribute(j, "aa")<<endl;
                }
	}
	cout << CYCLE_NUM << "\t" << cellEnv.getCurrentCellNum() << endl;
}

int main()
{
	try
	{
		cudaSetDevice(0);
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 100*1024*1024);

		struct timeval start, end;
        	int milliseconds;

		gettimeofday(&start, NULL);

		work();
		
		gettimeofday(&end, NULL);

		milliseconds = ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000.0) + 0.5;

		printf("Time Calc: %.2fs\n", (double)milliseconds/1000.0);
	}
	catch (ParaCellsError e)
	{
		cout << e.getMessage() << endl;
	}

	return 0;
}
