#ifndef UAVVLASERRANGEFINDER_H
#define UAVVLASERRANGEFINDER_H

#include "../UavvPacket.h"
class LaserRange{
public:
    unsigned int Length = 6;
    unsigned char StatusFlag;
    unsigned char Confidence;
    unsigned int Range = 0;
    LaserRange();
    ~LaserRange();
    static ParseResult TryParse(GimbalPacket packet, LaserRange *Packet);
    GimbalPacket Encode();
};
#endif
