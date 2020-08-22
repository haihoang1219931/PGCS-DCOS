#ifndef UAVSETFOCUS_H
#define UAVSETFOCUS_H

#include"../UavvPacket.h"

class UavvSetFocus
{
public:
    unsigned int Length = 2;
	unsigned short FocusPosition;

    UavvSetFocus();
    UavvSetFocus(unsigned short focusPosition);
    ~UavvSetFocus();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetFocus *SetFocus);
};

#endif
