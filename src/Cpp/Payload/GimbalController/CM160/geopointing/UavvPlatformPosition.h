#ifndef UAVVPLATFORMPOSITION_H
#define UAVVPLATFORMPOSITION_H

#include "../UavvPacket.h"

class UavvPlatformPosition
{
public:
    unsigned int Length = 10;
    float Latitude;
    float Longtitude;
    float Altitude;
    UavvPlatformPosition();
    ~UavvPlatformPosition();
    UavvPlatformPosition(float latitude,
                         float longtitude,
                         float altitude);
    static ParseResult TryParse(GimbalPacket packet, UavvPlatformPosition *PlatformPosition);
    GimbalPacket Encode();

};
#endif
