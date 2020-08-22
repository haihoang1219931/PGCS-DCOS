#ifndef UAVSETFFCTEMPERATUREDELTA_H
#define UAVSETFFCTEMPERATUREDELTA_H

#include"../UavvPacket.h"

class UavvSetFFCTemperatureDelta
{
public:
    unsigned int Length = 2;
	float TemperatureDelta;

	void setTemperatureDelta(float dt)
	{
		TemperatureDelta = dt;
    }

	float getTemperatureDelta(){
		return TemperatureDelta;
    }

    UavvSetFFCTemperatureDelta();
    UavvSetFFCTemperatureDelta(float t);
    ~UavvSetFFCTemperatureDelta();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetFFCTemperatureDelta *SetFFCTemperatureDelta);
};

#endif
