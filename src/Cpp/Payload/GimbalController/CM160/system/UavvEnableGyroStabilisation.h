#ifndef UAVVENABLEGYROSTABILISATION_H
#define UAVVENABLEGYROSTABILISATION_H

#include "../UavvPacket.h"

class UavvEnableGyroStabilisation
{
public:
    unsigned int Length = 2;
    unsigned char PanFlag;
    unsigned char TiltFlag;
    UavvEnableGyroStabilisation(unsigned char panFlag,unsigned char tiltFlag);
	UavvEnableGyroStabilisation();
	~UavvEnableGyroStabilisation();
	static ParseResult TryParse(GimbalPacket packet, UavvEnableGyroStabilisation *enableGyroGimbalPacket);
	GimbalPacket Encode();
};
#endif
