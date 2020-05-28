#ifndef UAVSETEOOPTICALZOOMPOSITION_H
#define UAVSETEOOPTICALZOOMPOSITION_H

#include"../UavvPacket.h"

class UavvSetEOOpticalZoomPosition
{
public:
    unsigned int Length = 2;
	unsigned short Position;

    UavvSetEOOpticalZoomPosition();
    UavvSetEOOpticalZoomPosition(unsigned short position);
    ~UavvSetEOOpticalZoomPosition();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetEOOpticalZoomPosition *SetEOOpticalZoomPosition);
};

#endif
