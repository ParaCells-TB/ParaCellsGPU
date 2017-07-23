#ifndef IDENTIFIERS_CUH
#define IDENTIFIERS_CUH

class Identifiers
{
private:
	char **h_identifiers;
	char **d_identifiers;
	int maxIdentifiersNum;
	int currentIdentifiersNum;

	//Flag
	bool hasUnpushedChangesInHost;
	bool hasUnpulledChangesInDevice;

public:
	Identifiers(int maxIdentifiersNum);
	virtual ~Identifiers();

	char **getHostIdentifiers();
	char **getDeviceIdentifiers();
	int getMaxIdentifiersNum();

	void setCurrentIdentifiersNum(int value);
	int getCurrentIdentifiersNum();

	void addIdentifier(int index, const char *identifierName);
	int findIdentifier(const char *identifierName);
};

#endif