#ifndef UAVVSETPANPOSITION_H
#define UAVVSETPANPOSITION_H

#include "../UavvPacket.h"
#include "../UavvPacketHelper.h"

class UavvSetPanPosition
{
public:
    unsigned int Length = 2;
    float PanPosition;
    UavvSetPanPosition();
    ~UavvSetPanPosition();
    UavvSetPanPosition(float panPosition);
    static ParseResult TryParse(GimbalPacket packet, UavvSetPanPosition *setPanPositionPacket);
	GimbalPacket Encode();
};

#endif
