#ifndef UAVDEFOG_H
#define UAVDEFOG_H

#include"../UavvPacket.h"

class UavvDefog
{
public:
    unsigned int Length = 2;
    FlagFog Flag;
    StrengthFog Strength;
    UavvDefog();
    UavvDefog(FlagFog flag, StrengthFog strength);
    ~UavvDefog();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvDefog *Defog);
};

#endif
