#ifndef UAVVSETPANTILTPOSITION_H
#define UAVVSETPANTILTPOSITION_H

#include "../UavvPacket.h"
class UavvSetPanTiltPosition
{
public:
    float PanPosition;
    float TiltPosition;
    unsigned int Length = 4;
    UavvSetPanTiltPosition();
    ~UavvSetPanTiltPosition();
    UavvSetPanTiltPosition(float panPosition, float tiltPosition);
    static ParseResult TryParse(GimbalPacket packet, UavvSetPanTiltPosition *setPanTiltPositionPacket);
	GimbalPacket Encode();
};
#endif // UAVVSETPANTILTPOSITION_H
