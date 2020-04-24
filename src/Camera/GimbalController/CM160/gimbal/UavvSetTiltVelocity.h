#ifndef UAVVSETTILTVELOCITY_H
#define UAVVSETTILTVELOCITY_H
#include "../UavvPacket.h"

class UavvSetTiltVelocity
{
public:
    float TiltVelocity;
    unsigned int Length = 2;
    UavvSetTiltVelocity();
    ~UavvSetTiltVelocity();
    UavvSetTiltVelocity(float tiltVelocity);
    static ParseResult TryParse(GimbalPacket packet, UavvSetTiltVelocity *setTiltPositionPacket);
	GimbalPacket Encode();
};
#endif
