#ifndef UAVVGIMBALORIENTATIONOFFSET_H
#define UAVVGIMBALORIENTATIONOFFSET_H

#include "../UavvPacket.h"

class UavvGimbalOrientationOffset
{
public:
    unsigned int Length = 7;
    unsigned char Reserved = 0;
    float Roll, Pitch, Yaw;

public:
	UavvGimbalOrientationOffset();
	~UavvGimbalOrientationOffset();
    UavvGimbalOrientationOffset(float roll, float pitch, float yaw);
	static ParseResult TryParse(GimbalPacket packet, UavvGimbalOrientationOffset *GimbalOrientationOffset);
    GimbalPacket Encode();
};
#endif
