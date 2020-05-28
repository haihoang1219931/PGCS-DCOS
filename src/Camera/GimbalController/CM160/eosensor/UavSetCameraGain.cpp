#include<iostream>
#include"UavSetCameraGain.h"

UavvSetCameraGain::UavvSetCameraGain(){}

UavvSetCameraGain::UavvSetCameraGain(unsigned char cameraGain)
{
    CameraGain = cameraGain;
};

UavvSetCameraGain::~UavvSetCameraGain(){}

GimbalPacket UavvSetCameraGain::Encode()
{
	unsigned char data[2];
    data[0] = Reserved;
    data[1] = CameraGain;
    return GimbalPacket(UavvGimbalProtocol::SetCameraGain, data, sizeof(data));
}

ParseResult UavvSetCameraGain::TryParse(GimbalPacket packet, UavvSetCameraGain *SetCameraGain)
{
    if (packet.Data.size() < SetCameraGain->Length)
	{
        return ParseResult::InvalidLength;
	}
	unsigned char gain;
	if (packet.Data[1]<0x16)
	{
		gain = packet.Data[1];
        *SetCameraGain = UavvSetCameraGain(gain);
        return ParseResult::Success;
	}
    return ParseResult::InvalidData;
};
