#ifndef UAVSETIRZOOM_H
#define UAVSETIRZOOM_H

#include "../UavvPacket.h"

class UavvSetIRZoom
{
public:
    unsigned int Length = 2;
	ZoomFlag Flag;
    unsigned char Reserved = 0;
    UavvSetIRZoom();
    UavvSetIRZoom(ZoomFlag flag);
    ~UavvSetIRZoom();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRZoom *SetIRZoom);
};

#endif
