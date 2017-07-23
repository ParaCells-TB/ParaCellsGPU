#include "ParaCellsError.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

ParaCellsError::ParaCellsError()
{
	message = NULL;
}

ParaCellsError::ParaCellsError(const char *message, const char *file, int line, int isFromCUDA)
{
	const char *err;
	if (isFromCUDA)
	{
		err = "[ParaCells Error][CUDA] ";
	}
	else
	{
		err = "[ParaCells Error] ";
	}

	char line_str[10];
	//itoa(line, line_str, 10);
	sprintf(line_str, "%d", line);
	this->message = (char *)malloc(sizeof(char)*(strlen(err) + strlen(file) + strlen(line_str) + strlen(message) + 6));
	strcpy(this->message, err);
	strcat(this->message, file);
	strcat(this->message, "(");
	strcat(this->message, line_str);
	strcat(this->message, "): ");
	strcat(this->message, message);
}

char* ParaCellsError::getMessage()
{
	return message;
}

ParaCellsError::~ParaCellsError()
{
	if (message)
	{
		//free(message);
	}
}
