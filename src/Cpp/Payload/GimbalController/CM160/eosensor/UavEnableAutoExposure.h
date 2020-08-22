#ifndef UAVENABLEAUTOEXPOSURE_H
#define UAVENABLEAUTOEXPOSURE_H

#include"../UavvPacket.h"

class UavvEnableAutoExposure
{
public:
    unsigned int Length = 2;
	ExposureStatus Status;

    UavvEnableAutoExposure();
    UavvEnableAutoExposure(bool enable);
    UavvEnableAutoExposure(ExposureStatus status);
    ~UavvEnableAutoExposure();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableAutoExposure *EnableAutoExposure);
};

#endif
