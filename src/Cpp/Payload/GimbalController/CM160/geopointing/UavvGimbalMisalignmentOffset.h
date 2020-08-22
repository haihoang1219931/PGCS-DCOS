#ifndef UAVVGIMBALMISALIGNMENTOFFSET_H
#define UAVVGIMBALMISALIGNMENTOFFSET_H

#include "../UavvPacket.h"

class UavvGimbalMisalignmentOffset
{
public:
    unsigned int Length = 43;
    unsigned char MountType;
    unsigned char Reserved[38];
    unsigned short Tilt, Pan;

    UavvGimbalMisalignmentOffset();
	~UavvGimbalMisalignmentOffset();
    UavvGimbalMisalignmentOffset(unsigned char mountType);
	static ParseResult TryParse(GimbalPacket packet, UavvGimbalMisalignmentOffset *GimbalMisalignmentOffset);
    GimbalPacket Encode();
};
#endif
