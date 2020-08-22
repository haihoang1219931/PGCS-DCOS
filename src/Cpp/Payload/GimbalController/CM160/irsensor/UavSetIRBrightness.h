#ifndef UAVSETIRBRIGHTNESS_H
#define UAVSETIRBRIGHTNESS_H

#include "../UavvPacket.h"

class UavvSetIRBrightness
{

public:
    unsigned int Length = 2;
	unsigned short Brightness;
    UavvSetIRBrightness();
    UavvSetIRBrightness(unsigned short brightness);
    ~UavvSetIRBrightness();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRBrightness *SetIRBrightness);
};

#endif
