#ifndef UAVSETIRPALETTE_H
#define UAVSETIRPALETTE_H

#include "../UavvPacket.h"

class UavvSetIRPalette
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
	PaletteMode Mode;

	void setMode(PaletteMode mode)
	{
		Mode = mode;
    }

	PaletteMode getMode(){
		return Mode;
    }

public:
    UavvSetIRPalette();
    UavvSetIRPalette(PaletteMode mode);
    ~UavvSetIRPalette();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRPalette *SetIRPalette);
};

#endif
