#ifndef UAVPERFORMFFC_H
#define UAVPERFORMFFC_H

#include "../UavvPacket.h"

class UavvPerformFFC
{

public:
    unsigned int Length = 2;
    unsigned char Reserved = 0x00;
    unsigned char PerformFFC = 0x01;
    UavvPerformFFC();
    ~UavvPerformFFC();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvPerformFFC *PerformFFC);
};

#endif
