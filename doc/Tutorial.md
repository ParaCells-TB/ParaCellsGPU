#ParaCells - Tutorial

##Create a ParaCells project

To create a new ParaCells project, just copy all the files in the ```src/``` directory into a new folder.
Then the new directory can be temporarily considered as an individual ParaCells project.

##Project file structure

The file structure of a ParaCells project is:

```
.
├── /bin
|   ├── run
├── Makefile
├── Base.cuh
├── Base.cu
├── BuiltInKernels.cuh
├── ...
├── Main.cu
└── CustomFunctions.cuh
```

* The ```/bin``` directory are runtime files for compiling and running. It may not be exist at the beginning.
* The ```Makefile``` is the execution script for project compilation, so that there is no need for you to look at or modify it.
* The ```Main.cu``` and ```CustomFunctions.cuh``` is the place where user implements their models for simulation. **[These are the only files needed to be modified]**
* The rest of files, such as ```Base.cuh``` and ```Base.cu``` are all library files, they should not be modified as well.

##Compile ParaCells project

When finish coding, instructions below could be used to compile the project.

1. Open Windows Command Prompt (CMD);
2. Located in the directory where the project is placed;
3. Execute command:

	```
	$ paracells compile
	```

> Logs of the compiling process would be printed into the prompt, including warnings and errors.

##Run ParaCells project

**Only projects that compiled successfully could be executed.**

To run a ParaCells project for computing or simulation, follow instructions:

1. Open Windows Command Prompt (CMD);
2. Located in the directory where the project is placed;
3. Execute command:

	```
	$ paracells run
	```

##Clean ParaCells project

If you want to remove the previous build, instructions below could be used to clean the project.

1. Open Windows Command Prompt (CMD);
2. Located in the directory where the project is placed;
3. Execute command:

	```
	$ paracells clean
	```

##Core concepts in ParaCells

###Concepts

In ParaCells library, there are 5 core concepts used in developing process. These concepts (objects) can be controled to implement your own models.

![](img/concepts.png)

* **Cell Environment** is a global concept of an environment. You can consider it as a virtual laboratory environment. We put a number of cells inside this environment, while each cell has several attributes. Between cells, the environment itself has some status as well, called environment parameters. Cells' attribute could be affected by environment, or it can affect the environment in turn. Of course, inside a program, it is possible to have more than one "Cell environment" simultaneously.
* **Cell** represents a single cell inside "Cell environment", it has several numerical attributes named "Cell attribute".
* **Cell Attribute** is a single numerical value, indicating a kind of attribute of a cell. (Only single-precision floating point value supported now)
* **Environment** represents extra-cellular environment between individual cells, it has several numerical attributes named "Environment attribute".
* **Environment Parameter** is a single numerical value, indicating a kind of attribute of the environment space. (Only single-precision floating point value supported now)

> In a "Cell environment", all cells should have the same kinds of attributes.

###Object-oriented programming

ParaCells library exposes functionalities in an object-oriented way. Concepts listed above are all classes or properties in the code level.

"Cell environment" corresponds to a class "CellEnv" in the ParaCells library. It has all the functionalities to control this "virtual laboratory", ranging from add cells, "Cell attributes", "Environment attributes" to set or get "Cell attribute", even parallelly update all cells status. Full methods and functionalities are listed in later sections.

"Cell" and "Environment" corresponds to classes with the same name in the ParaCells library. Objects of class "Cell" usually repersents one cell in "Cell environment", providing functionalities of get or set "Cell attributes", make the cell die or proliferate, etc. Objects of class "Environment" deontes the whole environment spaces in "Cell environment", providing functionalities of get or set "Environment attributes", generate uniform or poisson distributed random numbers, etc.

Concepts "Cell attribute" and "Environment attribute" are mainly properties inside those classes, so you can obtain or modify them through methods provided.

> "CellEnv" class can only be used in ```Main.cu```, while "Cell" and "Environment" classes can only be accessed through parameters of ```Custom functions``` in ```CustomFunctions.cuh```.
> 
> Details would be discussed in later sections.

##Start developing with ParaCells from demo project

Every new ParaCells project is shipped with a simple demo project. Understanding those code would be helpful for further development.

In a ParaCells project, ```Main.cu``` and ```CustomFunctions.cuh``` are the two files store your simulation program.

* **Main.cu**: The main program, running on CPU. All of your code should be written into this file except for the parallel functions. It has the ability to call functions in ```CustomFunctions.cuh``` in order to update the status of all the cells in parallel.
* **CustomFunctions.cuh**: a collection of functions, which perform operations on every single cell. These functions would be applied to all the cells automatically in parallel during simulation. Functions in this file can be invoked from ```Main.cu```.

###CustomFunctions.cuh

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

###Main.cu

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

###Compile and run

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

##Available classes and methods

###Main.cu

####[Class: CellEnv]

* **CellEnv(int maxCellNum, int maxAttributeNum, int maxEnvironmentAttributeNum)**

	This is the constructor of the class CellEnv, returning an instance of CellEnv.
	
	* maxCellNum: possible maximum number of cells in "Cell environment";
	* maxAttributeNum: possible maximum number of attributes in a single cell;
	* maxEnvironmentAttributeNum: possible maximum number of environment attributes in "Cell environment".
	
* **int getMaxCellNum()**

	The function returns the possible maximum number of cells in "Cell environment".

* **int getMaxCellAttributeNum()**

	The function returns the possible maximum number of attributes in a single cell.

* **int getMaxEnvironmentAttributeNum()**

	The function returns the possible maximum number of environment attributes in "Cell environment".

* **int getCurrentCellNum()**

	The function returns the current number of cells in "Cell environment".

* **int getCurrentAttributeNum()**

	The function returns the current number of attributes in a single cell.

* **int getCurrentEnvironmentAttributeNum()**

	The function returns the current number of environment attributes in "Cell environment".

* **void addCells(int num, int funcIndex = -1)**

	The function adds a certain number of cells into "Cell environment". The second parameter "funcIndex" is optional. When ignoring or passing -1 to it, attributes of newly added cells would be set to 0. Otherwise, "funcIndex" denotes the index of "Custom functions" which is used to initialize the newly added cells.
	
	> Only newly added cells would be affected in "funcIndex" specifing "Custom function".
	
	* num: number of cells added;
	* [optional] funcIndex: index of "Custom function" for newly added cells' initialization.

* **void addAttribute(char \*attrName, int funcIndex = -1)**

	The function adds a attribute to all cells in "Cell environment", and future added cells will have the attribute as well. The second parameter "funcIndex" is optional. When ignoring or passing -1 to it, newly added attribute of all existing cells would be set to 0. Otherwise, "funcIndex" denotes the index of "Custom functions" which is used to initialize the newly added attribute of all existing cells.
	
	> You can set other attributes in the "Custom funciton" that used in this function, but it is not recommended in order to keep the code understandable.
	
	* attrName: name of the added "Cell attribute";
	* [optional] funcIndex: index of "Custom function" for the added attribute initialization.

* **void addEnvironmentAttribute(char \*attrName, float value = 0)**

	The function adds a "Environment attribute" into "Cell environment". The second parameter "value" is used to initialize the added "Environment attribute", while it is optional. When ignoring it, default value 0 would be assigned.
	
	* paramName: name of the added "Environment attribute";
	* [optional] value: initial value of the added "Environment attribute".

* **void updateAllCells(int funcIndex = -1)**

	The function uses "Custom function" to update all cells' status in "Cell environment". The parameter "funcIndex" is optional. When ignoring or passing -1 to it, nothing would happen to the existing cells. Otherwise, "funcIndex" denotes the index of "Custom functions" which is used to update cells' status in parallel.
	
	* funcIndex: index of "Custom function" for updating existing cells' status.

* **void updateAllCellsWithoutFateDecision(int funcIndex = -1)**

	The function uses "Custom function" to update all cells' status in "Cell environment". The parameter "funcIndex" is optional. When ignoring or passing -1 to it, nothing would happen to the existing cells. Otherwise, "funcIndex" denotes the index of "Custom functions" which is used to update cells' status in parallel.
	
	* funcIndex: index of "Custom function" for updating existing cells' status.

	> This function is a simplified version of ```updateAllCells()```. The only difference between then is that "Custom Functions" running in this version cannot handle cells' death and proliferation (change in system size), so that it can provide higher computational performance.

* **void setCellAttribute(int cellId, char \*attrName, float value)**

	The function is used to update an attribute's value of a specific cell, indicated by its cell ID.
	
	> Cell ID ranges from 0 to (```getCurrentCellNum()``` - 1);
	> 
	> When calling ```updateAllCells()```, if ```die()``` or ```proliferate()``` is used, which means the total number of cells in "Cell environment" changes, ID of all cells would be changed automatically. It is not recommended to heavily depend on the unstable cell ID property.
	
	* cellId: ID property of a cell;
	* attrName: name of the modified "Cell attribute";
	* value: updated value.

* **void setEnvironmentAttribute(char \*attrName, float value)**

	The function is used to update an "Environment attribute" value.
	
	* attrName: name of the modified "Environment attribute";
	* value: updated value.

* **float getCellAttribute(int cellId, const char \*attrName)**

	The function is used to get an attribute's value of a specific cell, indicated by its cell ID.
	
	> Cell ID ranges from 0 to (```getCurrentCellNum()``` - 1);
	> 
	> When calling ```updateAllCells()```, if ```die()``` or ```proliferate()``` is used, which means the total number of cells in "Cell environment" changes, ID of all cells would be changed automatically. It is not recommended to heavily depend on the unstable cell ID property.
	
	* cellId: ID property of a cell;
	* attrName: name of the "Cell attribute";

* **float getEnvironmentAttribute(char \*attrName)**

	The function is used to get an "Environment attribute" value.
	
	* attrName: name of the "Environment attribute".

* **float sumAttributeFromAllCells(char \*attrName)**

	The function sums a specific "Cell attribute" of all existing cells, then returns the result.
	
	* attrName: name of the "Cell attribute".

* **void printCells()**

	The function prints all cells and their "Cell attributes" to the prompt.

* **void printEnvironmentAttributes()**

	The function prints all "Environment attributes" in "Cell Environment" to the prompt.

* **void print()**

	The function prints all "Environment attributes" and all cells' "Cell attributes" to the prompt.
	
	It is a combination of function ```printEnvironmentAttributes()``` and ```printCells()```.

###CustomFunctions.cuh

####[Class: Cell]

* **int getCellId()**

	The function returns ID property of the current cell.
	
	> Cell ID ranges from 0 to (```getCurrentCellNum()``` - 1);
	> 
	> When calling ```updateAllCells()```, if ```die()``` or ```proliferate()``` is used, which means the total number of cells in "Cell environment" changes, ID of all cells would be changed automatically. It is not recommended to heavily depend on the unstable cell ID property.

* **void setAttribute(char \*attrName, float value)**

	The function is used to set an attribute value of the current cell.
	
	* attrName: name of the modified "Cell attribute";
	* value: updated value.

* **float getAttribute(char \*attrName)**

	The function is used to get an attribute value of the current cell.
	
	* attrName: name of the "Cell attribute".

* **void setCustomObject(ParaCellsObject \*pointer)**

	The function is used to store a "Custom Object" inside the current cell.
	
	* pointer: pointer of the "Custom Object".

* **ParaCellsObject \*getCustomObject()**

	The function is used to get the "Custom Object" stored in the current cell. NULL will be returned if there are no "Custom Object" stored in it.

* **void die()**

	The function marks the current cell to be die. Current cell will disapper from "Cell environment" after finishing the ongoing "Custom function".
	
	> After calling ```die()```, other methods in this "Cell" object would be non-functional.

* **void proliferate()**

	The function marks the current cell to proliferate. Current cell will become two daughter cells after finishing the ongoing "Custom function".
	
	> Two daughter cells's attribute will initially copy from it's mother. 
	>
	> If you want to modify them, function ```setFirstDaughterCellAttribute()``` and ```setSecondDaughterCellAttribute()``` could be used.

* **void setFirstDaughterCellAttribute(char \*attrName, float value)**

	> This function is functional only after the ```proliferate()``` having been called.
	
	The function is used to modify "Cell attributes" of the first daughter cell after proliferation.
	
	* attrName: name of the modified "Cell attributes";
	* value: updated value.

* **void setSecondDaughterCellAttribute(char \*attrName, float value)**

	> This function is functional only after the ```proliferate()``` having been called.
	
	The function is used to modify "Cell attributes" of the second daughter cell after proliferation.
	
	* attrName: name of the modified "Cell attributes";
	* value: updated value.

####[Class: Environment]

* **float getAttribute(const char \*attrName)**

	The function is used to get an "Environment attribute" from the "Cell environment".
	
	* attrName: name of the "Environment attribute".

* **float incrAttribute(const char \*attrName, float value)**

	The function is used to increase "Environment attribute" by value. The increase of attributes is atomic among all the parallel executing "Cells" (increase operations from different cells will be queued, and sequentially perform to the attribute).
	
	* attrName: name of the "Environment attribute";
	* value: increased value.

	> Modifying the value of "Environment attributes" from "Custom functions" is not encouraged. Because atomic operations are considerably harmful for simulation performance.
	> 
	> We suggest to modify them only by using "CellEnv" object in ```Main.cu```.

* **float getUniformRandom()**

	The function returns a uniform distributed random number ranging from [0, 1].

* **float getPossionRandom(float lambda)**

	The function returns a possion distributed random number according to the provided "lambda".
	
	* lambda: value of parameter "lambda" of possion distributed equation.

* **int getCurrentCellNum()**

	The function returns the current number of cells in "Cell environment".