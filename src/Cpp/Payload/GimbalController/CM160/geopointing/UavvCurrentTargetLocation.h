#ifndef UAVVCURRENTTARGETLOCATION_H
#define UAVVCURRENTTARGETLOCATION_H

#include "../UavvPacket.h"

class UavvCurrentTargetLocation
{
public:
    unsigned int Length = 17;
	unsigned char Flag;
    float Latitude, Longitude;
    unsigned short Reserved01;
    float SlantRange;
    unsigned short Reserved02;
	void setFlagCurrentTargetLocation(unsigned char flag)
	{
		Flag = flag;
	}

    void setLatitudeCurrentTargetLocation(int latitude)
	{
		Latitude = latitude;
	}

    void setLongitudeCurrentTargetLocation(int longitude)
	{
		Longitude = longitude;
	}

    void setSlantRangeCurrentTargetLocation(unsigned int slantRange)
	{
		SlantRange = slantRange;
	}

	unsigned char getFlagCurrentTargetLocation()
	{
		return Flag;
	}

    int getLatitudeCurrentTargetLocation()
	{
		return Latitude;
	}

    int getLongitudeCurrentTargetLocation()
	{
		return Longitude;
	}

    unsigned int getSlantRangeCurrentTargetLocation()
	{
		return SlantRange;
	}
public:
	UavvCurrentTargetLocation();
	~UavvCurrentTargetLocation();
    UavvCurrentTargetLocation(unsigned char flag, int latitude, int longitude, unsigned short slantRange);
	static ParseResult TryParse(GimbalPacket packet, UavvCurrentTargetLocation *CurrentTargetLocation);
    GimbalPacket Encode();
};
#endif
