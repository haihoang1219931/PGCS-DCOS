#ifndef UAVSETIRPLATEAULEVEL_H
#define UAVSETIRPLATEAULEVEL_H

#include "../UavvPacket.h"

class UavvSetIRPlateauLevel
{
public:
    unsigned int Length = 2;
	unsigned short Level;
	void setLevel(unsigned short level)
	{
		Level = level;
    }

	unsigned short getLevel(){
		return Level;
    }

    UavvSetIRPlateauLevel();
    UavvSetIRPlateauLevel(unsigned short level);
    ~UavvSetIRPlateauLevel();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRPlateauLevel *SetIRPlateauLevel);
};

#endif
