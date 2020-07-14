#ifndef ARMLASERDEVICE_H
#define ARMLASERDEVICE_H

#include "../UavvPacket.h"
class ArmLaserDevice
{
public:
    ArmLaserDevice();
    ~ArmLaserDevice();
    unsigned int Length = 2;
    unsigned char Reserved = 0x34;
    unsigned char Arm;
    static ParseResult TryParse(GimbalPacket packet, ArmLaserDevice *Packet);
    GimbalPacket Encode();
};

#endif // ARMLASERDEVICE_H
