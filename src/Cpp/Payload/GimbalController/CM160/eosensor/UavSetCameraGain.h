#ifndef UAVSETCAMERAGAIN_H
#define UAVSETCAMERAGAIN_H

#include<iostream>
#include"../UavvPacket.h"

class UavvSetCameraGain
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
	unsigned char CameraGain;
    UavvSetCameraGain();
    UavvSetCameraGain(unsigned char speed);
    ~UavvSetCameraGain();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvSetCameraGain *SetCameraGain);
};

#endif
