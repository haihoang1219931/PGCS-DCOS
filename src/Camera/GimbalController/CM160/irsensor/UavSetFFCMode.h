#ifndef UAVSETFFCMODE_H
#define UAVSETFFCMODE_H

#include "../UavvPacket.h"

class UavvSetFFCMode
{
public:
    unsigned int Length = 2;
	FFCMode Mode;

	void setMode(FFCMode mode)
	{
		Mode = mode;
    }

	FFCMode getMode(){
		return Mode;
    }

public:
    UavvSetFFCMode();
    UavvSetFFCMode(FFCMode mode);
    ~UavvSetFFCMode();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetFFCMode *SetFFCMode);
};

#endif
