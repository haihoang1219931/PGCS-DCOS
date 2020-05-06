#ifndef UAVSETIRITTMIDPOINT_H
#define UAVSETIRITTMIDPOINT_H

#include "../UavvPacket.h"

class UavvSetIRITTMidpoint
{
public:
    unsigned int Length = 2;
    unsigned char Data01 = 0x01;
	unsigned char Midpoint;

	void setMidpoint(unsigned char midpoint)
	{
		Midpoint = midpoint;
    }

	unsigned char getMidpoint(){
		return Midpoint;
    }

    UavvSetIRITTMidpoint();
    UavvSetIRITTMidpoint(unsigned char midpoint);
    ~UavvSetIRITTMidpoint();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRITTMidpoint *SetIRITTMidpoint);
};

#endif
