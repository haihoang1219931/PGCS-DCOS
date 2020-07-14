#include "UavvSetTiltVelocity.h"

UavvSetTiltVelocity::UavvSetTiltVelocity() {}
UavvSetTiltVelocity::~UavvSetTiltVelocity() {}

UavvSetTiltVelocity::UavvSetTiltVelocity(float tiltVelocity)
{
	TiltVelocity = tiltVelocity;
}

ParseResult UavvSetTiltVelocity::TryParse(GimbalPacket packet, UavvSetTiltVelocity *setTiltVelocity)
{
    if (packet.Data.size() < setTiltVelocity->Length)
	{
        return ParseResult::InvalidLength;
	}
    float tiltVelocity;
	tiltVelocity = UavvPacketHelper::PacketToVelocity(packet.Data[1], packet.Data[2]);
    *setTiltVelocity = UavvSetTiltVelocity(tiltVelocity);
    return ParseResult::Success;
}

GimbalPacket UavvSetTiltVelocity::Encode()
{
	unsigned char data[2];
    vector<unsigned char>tmp_TiltVelocity;
	tmp_TiltVelocity = UavvPacketHelper::VelocityToPacket(TiltVelocity);
	for (int i = 0; i < 2; i++)
	{
        data[0 + i] = tmp_TiltVelocity[i];
	}
	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::SetTiltVelocity, data, sizeof(data));
	return result;
}
