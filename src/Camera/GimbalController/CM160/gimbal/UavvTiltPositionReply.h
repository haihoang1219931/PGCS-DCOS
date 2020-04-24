#ifndef UAVVTILTPOSITIONREPLY_H
#define UAVVTILTPOSITIONREPLY_H

#include "../UavvPacket.h"
#include "../UavvPacketHelper.h"
class UavvTiltPositionReply{
public:
    float TiltVelocity;
    unsigned int Length = 2;
    UavvTiltPositionReply();
    ~UavvTiltPositionReply();
    UavvTiltPositionReply(float tiltVelocity);
    static ParseResult TryParse(GimbalPacket packet, UavvTiltPositionReply *tiltVelocityPacket);
    GimbalPacket Encode();
};
#endif // UAVVPANPOSITIONREPLY_H
