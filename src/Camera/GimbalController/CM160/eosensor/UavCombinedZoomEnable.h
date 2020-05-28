#ifndef UAVCOMBINEDZOOMENABLE_H
#define UAVCOMBINEDZOOMENABLE_H

#include<iostream>
#include"../UavvPacket.h"

enum class CombinedZoomStatus
{
	Disable,
	Enable
};

class UavvCombinedZoomEnable
{
public:
    unsigned int Length = 2;
	CombinedZoomStatus Status;
    UavvCombinedZoomEnable();
    UavvCombinedZoomEnable(bool enable);
    UavvCombinedZoomEnable(CombinedZoomStatus status);
    ~UavvCombinedZoomEnable();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvCombinedZoomEnable *CombinedZoomEnable);
};

#endif
