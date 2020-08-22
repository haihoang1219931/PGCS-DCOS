#ifndef UAVDISABLEINFRAREDCUTFILTER_H
#define UAVDISABLEINFRAREDCUTFILTER_H

#include"../UavvPacket.h"

class UavvDisableInfraredCutFilter
{
public:
    unsigned int Length = 2;
	InfraredCutStatus Status;
    UavvDisableInfraredCutFilter();
    UavvDisableInfraredCutFilter(bool disable);
    UavvDisableInfraredCutFilter(InfraredCutStatus status);
    ~UavvDisableInfraredCutFilter();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvDisableInfraredCutFilter *DisableInfraredCutFilter);
};

#endif
