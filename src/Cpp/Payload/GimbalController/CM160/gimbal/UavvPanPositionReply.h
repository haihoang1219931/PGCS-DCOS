#ifndef UAVVPANPOSITIONREPLY_H
#define UAVVPANPOSITIONREPLY_H

#include "../UavvPacket.h"
#include "../UavvPacketHelper.h"
class UavvPanPositionReply{
public:
    float PanPositionReply;
    unsigned int Length = 2;
    UavvPanPositionReply();
    ~UavvPanPositionReply();
    UavvPanPositionReply(float panVelocity);
    static ParseResult TryParse(GimbalPacket packet, UavvPanPositionReply *setPanTiltPositionPacket);
    GimbalPacket Encode();
};
#endif // UAVVPANPOSITIONREPLY_H
