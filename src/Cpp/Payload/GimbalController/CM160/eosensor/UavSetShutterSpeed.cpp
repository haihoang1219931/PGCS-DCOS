#include"UavSetShutterSpeed.h"

UavvSetShutterSpeed::UavvSetShutterSpeed(){}

UavvSetShutterSpeed::UavvSetShutterSpeed(unsigned char shutterSpeed)
{
    ShutterSpeed = shutterSpeed;
}

UavvSetShutterSpeed::~UavvSetShutterSpeed(){}

GimbalPacket UavvSetShutterSpeed::Encode()
{
	unsigned char data[2];
	data[0] = 0;
    data[1] = ShutterSpeed;
    return GimbalPacket(UavvGimbalProtocol::SetShutterSpeed, data, sizeof(data));
}

ParseResult UavvSetShutterSpeed::TryParse(GimbalPacket packet, UavvSetShutterSpeed *SetShutterSpeed)
{
    if (packet.Data.size() < SetShutterSpeed->Length)
	{
        return ParseResult::InvalidLength;
	}
	unsigned char speed;
	if (packet.Data[1]<0x16)
	{
		speed = packet.Data[1];
        *SetShutterSpeed = UavvSetShutterSpeed(speed);
        return ParseResult::Success;
	}
    return ParseResult::InvalidData;
};
