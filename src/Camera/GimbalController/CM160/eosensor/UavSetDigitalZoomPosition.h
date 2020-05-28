#ifndef UAVSETDIGITALZOOMPOSITION_H
#define UAVSETDIGITALZOOMPOSITION_H

#include"../UavvPacket.h"

class UavvSetDigitalZoomPosition
{
public:
    unsigned int Length = 2;
	unsigned short Position;

    UavvSetDigitalZoomPosition();
    UavvSetDigitalZoomPosition(unsigned short position);
    ~UavvSetDigitalZoomPosition();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetDigitalZoomPosition *SetDigitalZoomPosition);
};

#endif
