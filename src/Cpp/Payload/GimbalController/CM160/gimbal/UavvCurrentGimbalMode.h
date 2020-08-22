#ifndef UAVVCURRENTGIMBALMODE_H
#define UAVVCURRENTGIMBALMODE_H

#include "../UavvPacket.h"

class UavvCurrentGimbalMode
{
public:
    unsigned int Length = 2;
    unsigned char Reserved;
    unsigned char GimbalMode;
	UavvCurrentGimbalMode(CurrentGimbalMode gimbalmode);
	UavvCurrentGimbalMode();
	~UavvCurrentGimbalMode();
	static ParseResult TryParse(GimbalPacket packet, UavvCurrentGimbalMode *gimmode);
	GimbalPacket encode();

};
#endif
