#ifndef UAVVSETPANVELOCITY_H
#define UAVVSETPANVELOCITY_H
#include "../UavvPacket.h"

class UavvSetPanVelocity
{
public:
    float PanVelocity;
    unsigned int Length = 2;
    UavvSetPanVelocity();
    ~UavvSetPanVelocity();
    UavvSetPanVelocity(float panVelocity);
    static ParseResult TryParse(GimbalPacket packet, UavvSetPanVelocity *setPanTiltPositionPacket);
	GimbalPacket Encode();
};
#endif
