#include <iostream>

#include "CellEnv.cuh"
#include "ParaCellsError.cuh"

using namespace std;

void work()
{
	CellEnv cellEnv(50, 2, 1);

	//Add cell attributes
	cellEnv.addCellAttribute("ancestor");
	cellEnv.addCellAttribute("value");

	//Set threshold
	cellEnv.addEnvironmentAttribute("threshold", 0.5f);

	//Add cells and initialization
	cellEnv.addCells(20, 0);

	//Print all
	cellEnv.print();

	//Cell fate decision
	cellEnv.updateAllCells(1);

	//Print cells
	cellEnv.printCells();
}

int main()
{
	try
	{
		work();
	}
	catch (ParaCellsError e)
	{
		cout << e.getMessage() << endl;
	}

	return 0;
}
