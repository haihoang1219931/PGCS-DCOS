#ifndef UAVSETIRBRIGHTNESSBIAS_H
#define UAVSETIRBRIGHTNESSBIAS_H

#include"../UavvPacket.h"

class UavvSetIRBrightnessBias
{
public:
    unsigned int Length = 2;
	short Bias;

	void setBias(short bias)
	{
		Bias = bias;
    }

	short getBias(){
		return Bias;
    }

    UavvSetIRBrightnessBias();
    UavvSetIRBrightnessBias(short bias);
    ~UavvSetIRBrightnessBias();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRBrightnessBias *SetIRBrightnessBias);
};

#endif
