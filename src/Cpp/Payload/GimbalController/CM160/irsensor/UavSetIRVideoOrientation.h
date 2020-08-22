#ifndef UAVSETIRVIDEOORIENTATION_H
#define UAVSETIRVIDEOORIENTATION_H

#include"../UavvPacket.h"

class UavvSetIRVideoOrientation
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
    VideoOrientationMode Mode;

    void setMode(VideoOrientationMode mode)
	{
		Mode = mode;
    }

    VideoOrientationMode getMode(){
		return Mode;
    }

    UavvSetIRVideoOrientation();
    UavvSetIRVideoOrientation(VideoOrientationMode mode);
    ~UavvSetIRVideoOrientation();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetIRVideoOrientation *SetIRVideoOrientation);
};

#endif
