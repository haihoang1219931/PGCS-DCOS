#ifndef UAVVENABLEMESSAGEACK_H
#define UAVVENABLEMESSAGEACK_H

#include "../UavvPacket.h"

class UavvEnableMessageACK
{
public:
    unsigned int Length = 2;
    unsigned char Data01;
    unsigned char Data02;
public:
	UavvEnableMessageACK();
	~UavvEnableMessageACK();
    UavvEnableMessageACK(unsigned char data01,unsigned char data02);
	static ParseResult TryParse(GimbalPacket packet, UavvEnableMessageACK *Packet);
	GimbalPacket Encode();
};
#endif
