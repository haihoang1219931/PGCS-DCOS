#ifndef UAVGETZOOMPOSITION_H
#define UAVGETZOOMPOSITION_H

#include "../UavvPacket.h"

class UavvGetZoomPosition
{
public:
    unsigned int Length = 2;
    unsigned char Data01;
    unsigned char Data02;
    UavvGetZoomPosition();
    ~UavvGetZoomPosition();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvGetZoomPosition *GetZoomPosition);
};

#endif
