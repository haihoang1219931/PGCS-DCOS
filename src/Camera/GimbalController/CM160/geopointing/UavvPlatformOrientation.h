#ifndef UAVVPLATFORMORIENTATION_H
#define UAVVPLATFORMORIENTATION_H

#include "../UavvPacket.h"

class UavvPlatformOrientation
{
public:
    unsigned int Length = 6;
    float EulerRoll;
    float EulerPitch;
    float EulerYaw;

	UavvPlatformOrientation();
	~UavvPlatformOrientation();
    UavvPlatformOrientation(float eulerRoll, float eulerPitch, float eulerYaw);
	static ParseResult TryParse(GimbalPacket packet, UavvPlatformOrientation *PlatformOrientation);
    GimbalPacket Encode();
};
#endif
