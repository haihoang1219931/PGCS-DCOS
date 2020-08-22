#ifndef UAVINVERTPICTURE_H
#define UAVINVERTPICTURE_H

#include"../UavvPacket.h"

class UavvInvertPicture
{
public:
    unsigned int Length = 2;
	InvertMode Mode;

    UavvInvertPicture();
    UavvInvertPicture(bool invert);
    UavvInvertPicture(InvertMode mode);
    ~UavvInvertPicture();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvInvertPicture *InvertPicture);
};

#endif
