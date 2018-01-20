# Start Developing with ParaCells

Every new ParaCells project is shipped with a simple demo project. Understanding those code would be helpful for further development.

In a ParaCells project, ```Main.cu``` and ```CustomFunctions.cuh``` are the two files store your simulation program.

* **Main.cu**: The main program, running on CPU. All of your code should be written into this file except for the parallel functions. It has the ability to call functions in ```CustomFunctions.cuh``` in order to update the status of all the cells in parallel.
* **CustomFunctions.cuh**: a collection of functions, which perform operations on every single cell. These functions would be applied to all the cells automatically in parallel during simulation. Functions in this file can be invoked from ```Main.cu```.

### CustomFunctions.cuh

Firstly, we will look into the file ```CustomFunctions.cuh```:

```
#include "CustomFunction.cuh"

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

__device__ CustomFunction customFunctions[2] = { InitCells, CellFateDicision };

```
This file consists of two functions and a global variable definition of array.

Functions in this file are calld "Custom functions". They have almost the same function signature except for the function name. At the beginnning, ```__device__``` specifier indicates that this function is running parallelly on a GPU device. Functions' return type must be void due to the disability in returning value cross devices from GPU to CPU. Each "Custom function" has the same parameters list, ```Cell *cell``` and ```Environment *env```, representing current cell object the function controls and environment object, respectively.

> Both objects' methods may not be familier to you now. We will introduce them in later sections.

The first "Custom function" (InitCells) serves as a initializer of cells. Inside the funciton, we set the initial value of the cell. Attribute "ancestor" equals to the current cell's ID, and attribute "value" get a uniform distributed random number from the outer environment.

The second "Custom function" named "CellFateDicision" apparently decides every cell's fate according to both their attributes and environment status. There is a conditional statement inside the function checking whether the cell's "value" attribute is smaller than an environment attribute "threshold". If it is, the cell is sentenced to die, then it will disappear after finishing this function. Otherwise, the cell would proliferate into two daughter cells. Their "ancestor" attribute would not be changed, but each one could only get "a half" of the value from their mother cell (just like the mass property). After finishing the function, current cell would then become two daughter cells in the system.

At the bottom, a ```__device__``` specified array of type "CustomFunction" is defined for registering "Custom functions". Only functions that are registered in the array can be called from main program "Main.cu", otherwise the function can only be called by functions inside this file.

> Only registered "Custom funcions" can be called from Main.cu. Not registered functions can "serve" their ability to the registered ones.

### Main.cu

```Main.cu``` file contains the main logic of the simulation program, including invoking parallel "Custom functions":

```
#include <iostream>
#include "CellEnv.cuh"
#include "ParaCellsError.cuh"
using namespace std;

void work()
{
	CellEnv cellEnv(50, 2, 1);

	//Add cell attributes
	cellEnv.addAttribute("ancestor");
	cellEnv.addAttribute("value");

	//Set threshold
	cellEnv.addEnvironmentParameter("threshold", 0.5f);

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

```

Scanning through the file, there are two functions "main" and "work" in it.

The "main()" function is the entry point of the program. In this function, there is nothing but one statement ```work();``` inside the ```try``` block. The rest of this function is designed to catch errors and print them out. Thus, it is strongly recommended to use the same code structure in every ParaCells project.

Genuine program logic is located in function "work()". Firstly we declared the "Cell environment" object, defining that it can hold at most 50 cells with 2 "Cell attributes", and 1 "Environment attribute". Then we added two "Cell attributes" named "ancestor" and "value", with one "Environment attribute" named "threshold" whose initial value equals 0.5. After that, we added 20 cells by calling ```cellEnv.addCells(20, 0);```. Second parameter this function call denotes the index zero of "Custom functions" array inside the file ```CustomFunctions.cuh```. ParaCells would use the parameter to find the corresponding "Custom function" and use it to initialize newly added cells in parallel. ```cellEnv.print()``` and ```cellEnv.printCells()``` functions can be used to print the status of "Cell Environment" (using std::cout), details could be found in later sections. Between them, another parallel function call ```cellEnv.updateAllCells(1);``` has been used to update all cells status. Parameter "1" denotes the index one of "Custom functions" array, and it's our "CellFateDecision" function defined in ```CustomFunctions.cuh``` previously.

> You are supposed to leave "main()" as it is, and modify "work()" to create your own simulation program.

### Compile and Execution

After finishing editing these two files, we can compile and run the project for simulation.

Firstly, login to the workstation, then located in the directory where the project is placed.

To compile the project, execute the command below:

```
$ make
```

If there is no error message printed out, it indicates that your code compile successfully. So that we can use the command below to perform simulation.

```
$ make run
```

For our demo project, you may notice the output is similar to:

```
==========Environment Parameters (1)==========
Name	Value
threshold		0.500000

==========Cells (20)==========
Cell-ID	ancestor		value
0			0.000000		0.450830
1			1.000000		0.000387
2			2.000000		0.069933
3			3.000000		0.271628
4			4.000000		0.652305
...

==========Cells (16)==========
Cell-ID	ancestor		value
0			4.000000		0.326153
1			4.000000		0.326153
2			9.000000		0.491162
...
```

It is normal that output may varies each time you run the program. For the initial 20 cells, their values are obtained from the random number generator, which would affect the cell fate decision process.
