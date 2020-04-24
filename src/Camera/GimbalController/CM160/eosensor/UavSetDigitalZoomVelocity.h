#ifndef UAVSETDIGITALZOOMVELOCITY_H
#define UAVSETDIGITALZOOMVELOCITY_H

#include"../UavvPacket.h"

class UavvSetDigitalZoomVelocity
{
public:
    unsigned int Length = 2;
    ZoomStatus Status;
	unsigned char Velocity;

    UavvSetDigitalZoomVelocity();
    UavvSetDigitalZoomVelocity(ZoomStatus status, unsigned char velocity);
    ~UavvSetDigitalZoomVelocity();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetDigitalZoomVelocity *SetDigitalZoomVelocity);
};

#endif
