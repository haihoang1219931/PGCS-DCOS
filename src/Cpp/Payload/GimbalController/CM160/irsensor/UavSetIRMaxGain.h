#ifndef UAVSETIRMAXGAIN_H
#define UAVSETIRMAXGAIN_H

#include"../UavvPacket.h"

class UavvSetIRMaxGain
{
public:
    unsigned int Length = 2;
	unsigned short MaxGain;

	void setMaxGain(unsigned short gain)
	{
		MaxGain = gain;
    }

	unsigned short getMaxGain(){
		return MaxGain;
    }

    UavvSetIRMaxGain();
    UavvSetIRMaxGain(unsigned short gain);
    ~UavvSetIRMaxGain();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRMaxGain *SetIRMaxGain);
};

#endif
