#include<iostream>
#include"UavResetIRCamera.h"

UavvResetIRCamera::UavvResetIRCamera(){}

UavvResetIRCamera::~UavvResetIRCamera(){}

GimbalPacket UavvResetIRCamera::Encode()
{
	unsigned char data[2];
    data[0] = Data01;
    data[1] = Data02;
    return GimbalPacket(UavvGimbalProtocol::ResetIRCamera, data, sizeof(data));
}

ParseResult UavvResetIRCamera::TryParse(GimbalPacket packet, UavvResetIRCamera *ResetIRCamera)
{
    if (packet.Data.size() < ResetIRCamera->Length)
	{
        return ParseResult::InvalidLength;
	}

	if ((packet.Data[0] != 0x00) || (packet.Data[1] != 0x00))
        return ParseResult::InvalidData;
    *ResetIRCamera = UavvResetIRCamera();
    return ParseResult::Success;
}
