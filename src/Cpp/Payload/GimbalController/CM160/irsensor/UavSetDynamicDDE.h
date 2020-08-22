#ifndef UAVSETDYNAMICDDE_H
#define UAVSETDYNAMICDDE_H

#include "../UavvPacket.h"


class UavvSetDynamicDDE
{
public:
    unsigned int Length = 2;
	ManualDDEStatus Status;
	unsigned char Sharpness;

	void setStatus(ManualDDEStatus status)
	{
		Status = status;
    }

	void setSharpness(unsigned char sharpness)
	{
		Sharpness = sharpness;
	}

	ManualDDEStatus getStatus(){
		return Status;
    }

	unsigned char getSharpness()
	{
		return Sharpness;
	}

    UavvSetDynamicDDE();
    UavvSetDynamicDDE(ManualDDEStatus status, unsigned char sharpness);
    ~UavvSetDynamicDDE();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetDynamicDDE *SetDynamicDDE);
};

#endif
