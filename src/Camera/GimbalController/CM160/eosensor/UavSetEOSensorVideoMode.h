#ifndef UAVSETEOSENSORVIDEOMODE_H
#define UAVSETEOSENSORVIDEOMODE_H

#include"../UavvPacket.h"

class UavvSetEOSensorVideoMode
{

public:
    unsigned int Length = 2;
    unsigned char Reserved;
	unsigned char VideoMode;

    UavvSetEOSensorVideoMode();
    UavvSetEOSensorVideoMode(unsigned char data0, unsigned char mode);
    ~UavvSetEOSensorVideoMode();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetEOSensorVideoMode *SetEOSensorVideoMode);
};

#endif
