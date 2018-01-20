# API References

### Main.cu

#### [Class: CellEnv]

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

### CustomFunctions.cuh

#### [Class: Cell]

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

#### [Class: Environment]

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