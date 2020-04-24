#ifndef UAVSETIRIS_H
#define UAVSETIRIS_H

#include"../UavvPacket.h"

class UavvSetIris
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
    unsigned char FStopIndex;

    UavvSetIris();
    UavvSetIris(unsigned char fStopIndex);
    ~UavvSetIris();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIris *SetIris);
};

#endif
