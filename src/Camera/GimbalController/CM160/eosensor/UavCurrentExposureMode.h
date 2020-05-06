#ifndef UAVCURRENTEXPOSUREMODE_H
#define UAVCURRENTEXPOSUREMODE_H

#include"../UavvPacket.h"


class UavvCurrentExposureMode
{
public:
    unsigned int Length = 2;
    ExposureMode Mode;
	unsigned char Index;

    UavvCurrentExposureMode();
    UavvCurrentExposureMode(unsigned char index, ExposureMode mode);
    ~UavvCurrentExposureMode();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvCurrentExposureMode *CurrentExposureMode);
};

#endif
