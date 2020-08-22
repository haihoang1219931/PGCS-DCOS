#ifndef UAVSETIRAGCMODE_H
#define UAVSETIRAGCMODE_H

#include "../UavvPacket.h"

class UavvSetIRAGCMode
{
public:
    unsigned int Length = 2;
	AGCMode Mode;

    UavvSetIRAGCMode();
    UavvSetIRAGCMode(AGCMode mode);
    ~UavvSetIRAGCMode();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRAGCMode *SetIRAGCMode);
};

#endif
