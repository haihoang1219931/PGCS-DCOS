#ifndef UAVSETIRGAINMODE_H
#define UAVSETIRGAINMODE_H

#include "../UavvPacket.h"


class UavvSetIRGainMode
{
public:

    unsigned int Length = 2;
    unsigned char Data01 = 0x01;
    GainMode Mode;
	void setMode(GainMode mode)
	{
		Mode = mode;
    }

	GainMode getMode(){
		return Mode;
    }

    UavvSetIRGainMode();
    UavvSetIRGainMode(GainMode mode);
    ~UavvSetIRGainMode();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRGainMode *SetIRGainMode);
};

#endif
