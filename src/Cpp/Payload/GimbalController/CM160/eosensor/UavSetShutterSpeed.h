#ifndef UAVSETSHUTTERSPEED_H
#define UAVSETSHUTTERSPEED_H

#include"../UavvPacket.h"

class UavvSetShutterSpeed
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
	unsigned char ShutterSpeed;

    UavvSetShutterSpeed();
    UavvSetShutterSpeed(unsigned char speed);
    ~UavvSetShutterSpeed();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetShutterSpeed *SetShutterSpeed);
};

#endif
