#ifndef UAVVCURRENTGIMBALPOSITIONRATE_H
#define UAVVCURRENTGIMBALPOSITIONRATE_H

#include "../UavvPacket.h"
#include "../UavvPacketHelper.h"

class UavvCurrentGimbalPositionRate
{
public:
    float PanVelocity;
    float TiltVelocity;
    float PanPosition;
    float TiltPosition;
    unsigned int Length = 8;
    UavvCurrentGimbalPositionRate();
    ~UavvCurrentGimbalPositionRate();
    UavvCurrentGimbalPositionRate(float panPosition, float tiltPosition, float panVelocity, float tiltVelocity);
    static ParseResult TryParse(GimbalPacket packet, UavvCurrentGimbalPositionRate *result);
};
#endif
