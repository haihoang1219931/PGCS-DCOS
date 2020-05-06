#ifndef UAVSETIRCONTRAST_H
#define UAVSETIRCONTRAST_H

#include "../UavvPacket.h"

class UavvSetIRContrast
{
public:
    unsigned int Length = 2;
	unsigned char Contrast;

	void setContrast(unsigned char contrast)
	{
		Contrast = contrast;
    }

	unsigned char getContrast(){
		return Contrast;
    }

    UavvSetIRContrast();
    UavvSetIRContrast(unsigned char contrast);
    ~UavvSetIRContrast();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRContrast *SetIRContrast);
};

#endif
