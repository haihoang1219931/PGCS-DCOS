#ifndef LASERRANGESTART_H
#define LASERRANGESTART_H

#include "../UavvPacket.h"
class LaserRangeStart
{
public:
    unsigned int Length = 5;
    unsigned char Mesuaring;
    unsigned int Reserved = 0x0000;
    LaserRangeStart();
    ~LaserRangeStart();
    static ParseResult TryParse(GimbalPacket packet, LaserRangeStart *Packet);
    GimbalPacket Encode();
};

#endif // LASERRANGESTART_H
