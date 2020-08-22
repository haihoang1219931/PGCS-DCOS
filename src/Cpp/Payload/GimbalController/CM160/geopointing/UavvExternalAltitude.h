#ifndef UAVVEXTERNALALTITUDE_H
#define UAVVEXTERNALALTITUDE_H

#include "../UavvPacket.h"
class UavvExternalAltitude
{
public:
    unsigned char Length = 13;
    bool UseRoll = true;
    bool UsePitch = true;
    bool UseYaw = true;
    bool SaveRoll = true;
    bool SavePitch = true;
    bool SaveYaw = true;
    float Roll;
    float Pitch;
    float Yaw;
public:
    UavvExternalAltitude();
    virtual ~UavvExternalAltitude();
    static ParseResult TryParse(GimbalPacket packet, UavvExternalAltitude *ExternalAltitude);
    GimbalPacket Encode();
};

#endif // UAVVEXTERNALALTITUDE_H
