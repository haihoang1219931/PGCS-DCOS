#ifndef UAVSETEOOPTICALZOOMVELOCITY_H
#define UAVSETEOOPTICALZOOMVELOCITY_H

#include"../UavvPacket.h"


class UavvSetEOOpticalZoomVelocity
{
public:
    unsigned int Length = 2;
    ZoomStatus Status = ZoomStatus::Stop;
	unsigned char Velocity;

    UavvSetEOOpticalZoomVelocity();
    UavvSetEOOpticalZoomVelocity(ZoomStatus status, unsigned char velocity);
    ~UavvSetEOOpticalZoomVelocity();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetEOOpticalZoomVelocity *SetEOOpticalZoomVelocity);
};

#endif
