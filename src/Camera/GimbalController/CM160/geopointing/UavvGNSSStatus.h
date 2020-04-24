#ifndef UAVVGNSSSTATUS_H
#define UAVVGNSSSTATUS_H

#include "../UavvPacket.h"

class UavvGNSSStatus
{
public:
    unsigned int Length = 4;
    unsigned short SystemFlag, FilterFlag;

    void setSystemGNSSStatus(unsigned short flagSystem)
	{
		SystemFlag = flagSystem;
	}

    void setFilterGNSSStatus(unsigned short flagFilter)
	{
		FilterFlag = flagFilter;
	}

    unsigned short getSystemGNSSStatus()
	{
		return SystemFlag;
	}

    unsigned short getFilterGNSSStatus()
	{
		return FilterFlag;
	}
	UavvGNSSStatus();
	~UavvGNSSStatus();
    UavvGNSSStatus(unsigned short flagSystem, unsigned short flagFilter);
	static ParseResult TryParse(GimbalPacket packet, UavvGNSSStatus *GNSSStatus);
    GimbalPacket Encode();
};
#endif
