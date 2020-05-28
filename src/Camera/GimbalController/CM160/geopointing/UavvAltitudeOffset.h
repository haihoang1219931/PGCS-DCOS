#ifndef UAVVALTITUDEOFFSET_H
#define UAVVALTITUDEOFFSET_H

#include "../UavvPacket.h"

class UavvAltitudeOffset
{
public:
    unsigned int Length = 3;
    unsigned char Reserved = 0;
    short Altitude;

    void setAltitude(short altitude)
	{
		Altitude = altitude;
	}

    short getAltitude()
	{
		return Altitude;
    }
	UavvAltitudeOffset();
	~UavvAltitudeOffset();
    UavvAltitudeOffset(short altitude);
	static ParseResult TryParse(GimbalPacket packet, UavvAltitudeOffset *AltitudeOffset);
    GimbalPacket Encode();
};
#endif
