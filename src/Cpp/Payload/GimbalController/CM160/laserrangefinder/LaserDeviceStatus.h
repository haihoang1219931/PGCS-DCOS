#ifndef LASERDEVICESTATUS_H
#define LASERDEVICESTATUS_H

#include "../UavvPacket.h"
class LaserDeviceStatus
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0x51;
    unsigned char Info;
    LaserDeviceStatus();
    ~LaserDeviceStatus();
    static ParseResult TryParse(GimbalPacket packet, LaserDeviceStatus *Packet);
    GimbalPacket Encode();
};

#endif // LASERDEVICESTATUS_H
