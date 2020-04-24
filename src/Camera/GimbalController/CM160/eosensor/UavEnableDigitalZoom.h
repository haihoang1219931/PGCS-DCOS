#ifndef UAVENABLEDIGITALZOOM_H
#define UAVENABLEDIGITALZOOM_H

#include"../UavvPacket.h"

class UavvEnableDigitalZoom
{
public:
    unsigned int Length = 2;
	DigitalZoomStatus Status;
    UavvEnableDigitalZoom();
    UavvEnableDigitalZoom(bool enable);
    UavvEnableDigitalZoom(DigitalZoomStatus status);
    ~UavvEnableDigitalZoom();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableDigitalZoom *EnableDigitalZoom);
};

#endif
