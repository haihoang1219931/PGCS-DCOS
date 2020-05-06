#ifndef UAVENABLEMANUALIRIS_H
#define UAVENABLEMANUALIRIS_H

#include"../UavvPacket.h"


class UavvEnableManualIris
{
public:
    unsigned int Length = 2;
	ManualIrisStatus Status;

    UavvEnableManualIris();
    UavvEnableManualIris(bool enable);
    UavvEnableManualIris(ManualIrisStatus status);
    ~UavvEnableManualIris();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableManualIris *EnableManualIris);
};

#endif
