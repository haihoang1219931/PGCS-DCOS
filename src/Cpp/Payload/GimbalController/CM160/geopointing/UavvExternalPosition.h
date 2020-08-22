#ifndef UAVVEXTERNALPOSITION_H
#define UAVVEXTERNALPOSITION_H

#include "../UavvPacket.h"
class UavvExternalPosition
{
public:
    int Length = 13;
    unsigned char Flag = 0xff;
    float Latitude;
    float Longtitude;
    float Altitude;
public:
    UavvExternalPosition();
    virtual ~UavvExternalPosition();
    static ParseResult TryParse(GimbalPacket packet, UavvExternalPosition *ExternalPosition);
    GimbalPacket Encode();
};

#endif // UAVVEXTERNALPOSITION_H
