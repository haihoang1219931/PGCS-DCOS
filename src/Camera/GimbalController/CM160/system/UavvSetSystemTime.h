#ifndef UAVVSETSYSTEMTIME_H
#define UAVVSETSYSTEMTIME_H

#include "../UavvPacket.h"

class UavvSetSystemTime
{
public:
    unsigned int Length = 4;
	unsigned char Weekday;
	unsigned char Date;
	unsigned char Month;
	unsigned char Year;
    int second;
	UavvSetSystemTime(unsigned char weekday, unsigned char date, unsigned char month, unsigned char year);
	~UavvSetSystemTime();
	UavvSetSystemTime();
	static ParseResult TryParse(GimbalPacket packet, UavvSetSystemTime *SystemTime);
	GimbalPacket encode();
};
#endif
