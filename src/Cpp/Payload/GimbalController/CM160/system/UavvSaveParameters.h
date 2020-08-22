#ifndef UAVVSAVEPARAMETERS_H
#define UAVVSAVEPARAMETERS_H

#include "../UavvPacket.h"

enum class Parameters
{
    Save = 0x01,
    RestoreDefaults = 0x02,
};
class UavvSaveParameters
{
public:
    unsigned int Length = 2;
    unsigned char Data = 1;
	unsigned char Parameter;
	UavvSaveParameters(unsigned char parameter);
	~UavvSaveParameters();
	UavvSaveParameters();
	static ParseResult TryParse(GimbalPacket packet, UavvSaveParameters *parameter);
    GimbalPacket Encode();
};
#endif
