#include "UavvGimbalOrientationOffset.h"


UavvGimbalOrientationOffset::UavvGimbalOrientationOffset() {}
UavvGimbalOrientationOffset::~UavvGimbalOrientationOffset() {}

UavvGimbalOrientationOffset::UavvGimbalOrientationOffset(
        float roll,
        float pitch,
        float yaw)
{
    Roll = roll;
    Pitch = pitch;
    Yaw = yaw;
}

ParseResult UavvGimbalOrientationOffset::TryParse(GimbalPacket packet, UavvGimbalOrientationOffset*GimbalOrientationOffset)
{
    if (packet.Data.size() < GimbalOrientationOffset->Length)
	{
        return ParseResult::InvalidLength;
	}
    GimbalOrientationOffset->Reserved = packet.Data[0];
    short roll, pitch;
    unsigned short yaw;
    roll = ByteManipulation::ToInt16(packet.Data.data(),1,Endianness::Big);
    pitch = ByteManipulation::ToInt16(packet.Data.data(),3,Endianness::Big);
    yaw = ByteManipulation::ToUInt16(packet.Data.data(),5,Endianness::Big);
    GimbalOrientationOffset->Roll = (float)((float)(roll - 18000) / 100.0f + 180);
    GimbalOrientationOffset->Pitch = (float)((float)(pitch - 18000) / 100.0f + 180);
    GimbalOrientationOffset->Yaw = (float)((float)(yaw - 18000) / 100.0f + 180);
	return ParseResult::Success;
}

GimbalPacket UavvGimbalOrientationOffset::Encode()
{
	unsigned char data[7];
    data[0] = Reserved;
    ByteManipulation::ToBytes(Roll,Endianness::Big,data,1);
    ByteManipulation::ToBytes(Pitch,Endianness::Big,data,3);
    ByteManipulation::ToBytes(Yaw,Endianness::Big,data,5);
	return GimbalPacket(UavvGimbalProtocol::PlatformOrientationOffset, data, sizeof(data));
}
