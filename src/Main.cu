#include <iostream>

#include "CellEnv.cuh"
#include "ParaCellsError.cuh"
#include "TestAssets.cuh"

using namespace std;

int main()
{
	try
	{
		//initialTest();
		//cellDeathAndProliferationTest();
		//randomTest();
		//errorTest();
		//CUDAErrorTest();
		//incrEnvironmentAttributeTest();
		//customObjectTest();
	}
	catch (ParaCellsError e)
	{
		cout << e.getMessage() << endl;
	}

	return 0;
}
