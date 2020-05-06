#ifndef UAVENABLELENSSTABILIZATION_H
#define UAVENABLELENSSTABILIZATION_H

#include"../UavvPacket.h"


class UavvEnableLensStabilization
{
public:
    unsigned int Length = 2;
	LensStabilizationStatus Status;

    UavvEnableLensStabilization();
    UavvEnableLensStabilization(bool enable);
    UavvEnableLensStabilization(LensStabilizationStatus status);
    ~UavvEnableLensStabilization();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableLensStabilization *EnableLensStabilization);
};

#endif
