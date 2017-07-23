#ifndef TESTASSETS_CUH
#define TESTASSETS_CUH

#include <iostream>

#include "CellEnv.cuh"

using namespace std;

void displayCurrentNums(CellEnv &cellEnv)
{
	cout << "Cell Count: " << cellEnv.getCurrentCellNum() << endl;
	cout << "Cell Attribute Count: " << cellEnv.getCurrentCellAttributeNum() << endl;
	cout << "Env. Attribute Count: " << cellEnv.getCurrentEnvironmentAttributeNum() << endl;
}

void initialTest()
{
	cout << "=====200 cells, 5 cell attributes, 10 environment attributes=====" << endl;

	cout << endl << "=====After initialization=====" << endl;
	CellEnv cellEnv(200, 5, 10);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Add 50 cells=====" << endl;
	cellEnv.addCells(50);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Add cell attribute 'val1'=====" << endl;
	cellEnv.addCellAttribute("val1");
	displayCurrentNums(cellEnv);

	cout << endl << "=====Update cell-1's attribute 'val1' to 11=====" << endl;
	cout << "=====Update cell-2's attribute 'val1' to 22=====" << endl;
	cout << "=====Update cell-5's attribute 'val1' to 55=====" << endl;
	cellEnv.setCellAttribute(1, "val1", 11);
	cellEnv.setCellAttribute(2, "val1", 22);
	cellEnv.setCellAttribute(5, "val1", 55);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Get all cells' attribute 'val1'=====" << endl;
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val1: " << cellEnv.getCellAttribute(i, "val1") << endl;
	}

	cout << endl << "=====Get sum of all cells' attribute 'val1'=====" << endl;
	cout << "Sum val1: " << cellEnv.sumAttributeFromAllCells("val1") << endl;

	cout << endl << "=====Update all cells: attribute 'val1' plus 1=====" << endl;
	cellEnv.updateAllCellsWithoutFateDecision(0);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val1: " << cellEnv.getCellAttribute(i, "val1") << endl;
	}

	cout << endl << "=====Get sum of all cells' attribute 'val1'=====" << endl;
	cout << "Sum val1: " << cellEnv.sumAttributeFromAllCells("val1") << endl;

	cout << endl << "=====Add cell attribute 'val2', initialized to cellId=====" << endl;
	cellEnv.addCellAttribute("val2", 1);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val1: " << cellEnv.getCellAttribute(i, "val1") << "  val2: " << cellEnv.getCellAttribute(i, "val2") << endl;
	}

	cout << endl << "=====Add another 100 cells, initialized as above=====" << endl;
	cellEnv.addCells(100, 2);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val1: " << cellEnv.getCellAttribute(i, "val1") << "  val2: " << cellEnv.getCellAttribute(i, "val2") << endl;
	}

	cout << endl << "=====Get sum of all cells' attribute 'val1'=====" << endl;
	cout << "Sum val1: " << cellEnv.sumAttributeFromAllCells("val1") << endl;

	cout << endl << "=====Get sum of all cells' attribute 'val2'=====" << endl;
	cout << "Sum val2: " << cellEnv.sumAttributeFromAllCells("val2") << endl;

	cout << endl << "=====Add environment attribute 'env1'=====" << endl;
	cellEnv.addEnvironmentAttribute("env1");
	displayCurrentNums(cellEnv);

	cout << endl << "=====Update environment attribute 'env1' to 10=====" << endl;
	cellEnv.setEnvironmentAttribute("env1", 10);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Get environment attribute 'env1'=====" << endl;
	cout << "env1: " << cellEnv.getEnvironmentAttribute("env1") << endl;

	cout << endl << "=====Add environment attribute 'env2', initialized to 20=====" << endl;
	cellEnv.addEnvironmentAttribute("env2", 20);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Get environment attribute 'env2'=====" << endl;
	cout << "env2: " << cellEnv.getEnvironmentAttribute("env2") << endl;

	cout << endl << "=====Update all cells: attribute 'val1' to 'env1', attribute 'val2' to 'env2'=====" << endl;
	cellEnv.updateAllCellsWithoutFateDecision(3);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val1: " << cellEnv.getCellAttribute(i, "val1") << "  val2: " << cellEnv.getCellAttribute(i, "val2") << endl;
	}
}

void cellDeathAndProliferationTest()
{
	cout << "=====200 cells, 5 cell attributes, 10 environment attributes=====" << endl;

	cout << endl << "=====After initialization=====" << endl;
	CellEnv cellEnv(200, 5, 10);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Add 20 cells=====" << endl;
	cellEnv.addCells(20);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Add cell attribute 'val2'=====" << endl;
	cellEnv.addCellAttribute("val2", 1);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Get all cells' attribute 'val2'=====" << endl;
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val2: " << cellEnv.getCellAttribute(i, "val2") << endl;
	}

	cout << endl << "=====Update all cells: cellId%2=0 die, then cellId%3=0 proliferate=====" << endl;
	cellEnv.updateAllCells(4);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " val2: " << cellEnv.getCellAttribute(i, "val2") << endl;
	}
}

void randomTest()
{
	cout << "=====200 cells, 5 cell attributes, 10 environment attributes=====" << endl;

	cout << endl << "=====After initialization=====" << endl;
	CellEnv cellEnv(200, 5, 10);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Add 20 cells=====" << endl;
	cellEnv.addCells(20);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Add attribute 'randomVal'=====" << endl;
	cellEnv.addCellAttribute("randomVal");
	displayCurrentNums(cellEnv);

	cout << endl << "=====Get all cells' attribute 'randomVal'=====" << endl;
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " randomVal: " << cellEnv.getCellAttribute(i, "randomVal") << endl;
	}

	cout << endl << "=====Update all cells: generate uniform random values, and store them into 'randomVal'=====" << endl;
	cellEnv.updateAllCells(5);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " randomVal: " << cellEnv.getCellAttribute(i, "randomVal") << endl;
	}

	cout << endl << "=====[Second]Update all cells: generate uniform random values, and store them into 'randomVal'=====" << endl;
	cellEnv.updateAllCells(5);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " randomVal: " << cellEnv.getCellAttribute(i, "randomVal") << endl;
	}

	cout << endl << "=====Add attribute 'possionRandomVal'=====" << endl;
	cellEnv.addCellAttribute("possionRandomVal");
	displayCurrentNums(cellEnv);

	cout << endl << "=====Get all cells' attribute 'possionRandomVal'=====" << endl;
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " possionRandomVal: " << cellEnv.getCellAttribute(i, "possionRandomVal") << endl;
	}

	cout << endl << "=====Add environment attribute 'lambda' = 100=====" << endl;
	cellEnv.addEnvironmentAttribute("lambda", 100);
	displayCurrentNums(cellEnv);

	cout << endl << "=====Update all cells: generate possion random values, and store them into 'possionRandomVal'=====" << endl;
	cellEnv.updateAllCells(6);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " possionRandomVal: " << cellEnv.getCellAttribute(i, "possionRandomVal") << endl;
	}

	cout << endl << "=====[Second]Update all cells: generate possion random values, and store them into 'possionRandomVal'=====" << endl;
	cellEnv.updateAllCells(6);
	displayCurrentNums(cellEnv);
	for (int i = 0; i < cellEnv.getCurrentCellNum(); i++)
	{
		cout << "Cell-" << i << " possionRandomVal: " << cellEnv.getCellAttribute(i, "possionRandomVal") << endl;
	}
}

void errorTest()
{
	CellEnv cellEnv(100, 2, 2);

	cellEnv.addEnvironmentAttribute("param1");
	cellEnv.addEnvironmentAttribute("param2");
	//cellEnv.addEnvironmentAttribute("param3");

	cellEnv.addCellAttribute("attr1");
	cellEnv.addCellAttribute("attr2");
	cellEnv.addCellAttribute("attr3");

	cout << "Allocation successfully." << endl;
}

void CUDAErrorTest()
{
	CellEnv cellEnv(100000000, 100000000, 100000000);
}

void incrEnvironmentAttributeTest()
{
	CellEnv cellEnv(1000, 0, 1);

	cellEnv.addCells(1000);
	cellEnv.addEnvironmentAttribute("attr");
	cellEnv.updateAllCells(7);

	cout << "Expected Result: " << 499500 <<endl;
	cout << "Computed Result: " << cellEnv.getEnvironmentAttribute("attr") << endl;
	if (cellEnv.getEnvironmentAttribute("attr") == 499500)
	{
		cout << "[Success]" << endl;
	}
	else
	{
		cout << "[Fail]" << endl;
	}
}

void customObjectTest()
{
	CellEnv cellEnv(2000, 1, 0);

	cellEnv.addCells(1000);
	cellEnv.addCellAttribute("val");
	cellEnv.updateAllCells(8);
	cellEnv.updateAllCells(9);

	cout << cellEnv.sumAttributeFromAllCells("val") << " ";
	if (abs(cellEnv.sumAttributeFromAllCells("val") - 1000000) < 1e-5)
	{
		cout << "[Success]" << endl;
	}
	else
	{
		cout << "[Fail]" << endl;
	}
}

#endif
