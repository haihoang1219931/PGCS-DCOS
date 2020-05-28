#ifndef UAVVSTOWCONFIGURATION_H
#define UAVVSTOWCONFIGURATION_H
#include "../UavvPacket.h"

class UavvStowConfiguration
{
public:
    unsigned int Length = 8;
    unsigned char SaveFlash;
    unsigned char EnableAutoStow;
    unsigned short StowedTimeoutPeriod;
    unsigned short StowedPan;
    unsigned short StowedTilt;
    UavvStowConfiguration(unsigned char saveFlash,
                            unsigned char enableAutoStow,
                            unsigned short stowedTimeoutPeriod,
                            unsigned short stowedPan,
                            unsigned short stowedTilt);
	~UavvStowConfiguration();
	UavvStowConfiguration();
	static ParseResult TryParse(GimbalPacket packet, UavvStowConfiguration *stowConfiguration);
	GimbalPacket encode();
};
#endif // UAVVSTOWCONFIGURATION_H
