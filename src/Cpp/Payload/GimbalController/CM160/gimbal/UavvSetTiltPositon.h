#ifndef UAVVSETTILTPOSITION_H
#define UAVVSETTILTPOSITION_H
#include "../UavvPacket.h"
#include "../UavvPacketHelper.h"

class UavvSetTiltPosition
{
public:
    unsigned int Length = 2;
    float TiltPosition;
    UavvSetTiltPosition();
    ~UavvSetTiltPosition();
    UavvSetTiltPosition(float panPosition);
    static ParseResult TryParse(GimbalPacket packet, UavvSetTiltPosition *setTiltPositionPacket);
	GimbalPacket Encode();
};
#endif
