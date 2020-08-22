#ifndef UAVVPACKETSETPANTILTVELOCITY_H
#define UAVVPACKETSETPANTILTVELOCITY_H

#include "../UavvPacket.h"
#include "../UavvPacketHelper.h"
class UavvSetPanTiltVelocity
{
public:
    float PanVelocity;
    float TiltVelocity;
    unsigned int Length = 4;
    UavvSetPanTiltVelocity();
    ~UavvSetPanTiltVelocity();
    UavvSetPanTiltVelocity(float panVelocity, float tiltVelocity);
    static ParseResult TryParse(GimbalPacket packet, UavvSetPanTiltVelocity *setPanTiltPositionPacket);
	GimbalPacket Encode();
};
#endif
