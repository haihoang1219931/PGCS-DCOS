#include "UavvPlatformOrientation.h"


UavvPlatformOrientation::UavvPlatformOrientation() {}
UavvPlatformOrientation::~UavvPlatformOrientation() {}

UavvPlatformOrientation::UavvPlatformOrientation(float eulerRoll, float eulerPitch, float eulerYaw)
{
    EulerRoll = eulerRoll;
    EulerPitch = eulerPitch;
    EulerYaw = eulerYaw;
}

ParseResult UavvPlatformOrientation::TryParse(GimbalPacket packet, UavvPlatformOrientation*PlatformOrientation)
{
    if (packet.Data.size() < PlatformOrientation->Length)
	{
        return ParseResult::InvalidLength;
	}
    short roll, pitch;
    unsigned short yaw;
    roll = ByteManipulation::ToInt16(packet.Data.data(),0,Endianness::Big);
    pitch = ByteManipulation::ToInt16(packet.Data.data(),2,Endianness::Big);
    yaw = ByteManipulation::ToUInt16(packet.Data.data(),4,Endianness::Big);
    PlatformOrientation->EulerRoll = (float)((float)roll / 65535.0f * 360.0f);
    PlatformOrientation->EulerPitch = (float)((float)pitch / 65535.0f * 360.0f);
    PlatformOrientation->EulerYaw = (float)((float)yaw / 65535.0f * 360.0f);
	return ParseResult::Success;
}

GimbalPacket UavvPlatformOrientation::Encode()
{
    unsigned char data[6];
    int roll, pitch, yaw;
    roll = (EulerRoll * 65535.0f / 360.0f);
    pitch = (EulerPitch * 65535.0f / 360.0f);
    yaw = (EulerYaw  * 65535.0f / 360.0f);
    ByteManipulation::ToBytes((short)roll,Endianness::Big,data,0);
    ByteManipulation::ToBytes((short)pitch,Endianness::Big,data,2);
    ByteManipulation::ToBytes((unsigned short)yaw,Endianness::Big,data,4);
	return GimbalPacket(UavvGimbalProtocol::PlatformOrientation, data, sizeof(data));
}
