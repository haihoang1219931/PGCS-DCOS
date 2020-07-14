#ifndef UAVRESETIRCAMERA_H
#define UAVRESETIRCAMERA_H

#include"../UavvPacket.h"

class UavvResetIRCamera
{

public:
    unsigned int Length = 2;
    unsigned char Data01 = 0;
    unsigned char Data02 = 0;
    UavvResetIRCamera();
    ~UavvResetIRCamera();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvResetIRCamera *ResetIRCamera);
};

#endif
