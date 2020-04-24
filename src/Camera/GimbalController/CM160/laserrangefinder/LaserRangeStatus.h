#ifndef LASERRANGESTATUS_H
#define LASERRANGESTATUS_H

#include "../UavvPacket.h"
class LaserRangeStatus
{
public:
    unsigned int Length = 2;
    unsigned char Initialized;
    unsigned char RangeMode;
    LaserRangeStatus();
    ~LaserRangeStatus();
    static ParseResult TryParse(GimbalPacket packet, LaserRangeStatus *Packet);
    GimbalPacket Encode();
};

#endif // LASERRANGESTATUS_H
