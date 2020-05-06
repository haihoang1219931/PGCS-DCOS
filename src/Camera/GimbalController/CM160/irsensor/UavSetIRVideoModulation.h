#ifndef UAVSETIRVIDEOMODULATION_H
#define UAVSETIRVIDEOMODULATION_H

#include "../UavvPacket.h"

class UavvSetIRVideoModulation
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
	VideoModulation VideoModule;
    UavvSetIRVideoModulation();
    UavvSetIRVideoModulation(VideoModulation module);
    ~UavvSetIRVideoModulation();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRVideoModulation *SetIRVideoModulation);
};

#endif
