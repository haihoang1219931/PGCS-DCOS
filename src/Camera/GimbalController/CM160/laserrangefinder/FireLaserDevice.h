#ifndef FIRELASERDEVICE_H
#define FIRELASERDEVICE_H

#include "../UavvPacket.h"
class FireLaserDevice
{
public:
    unsigned int Length = 6;
    unsigned char Reserved = 0x34;
    unsigned char Fire;
    unsigned int VerificationSequence = 0xC5E70593;
    FireLaserDevice();
    ~FireLaserDevice();
    static ParseResult TryParse(GimbalPacket packet, FireLaserDevice *Packet);
    GimbalPacket Encode();
};

#endif // FIRELASERDEVICE_H
