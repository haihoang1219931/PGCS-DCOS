#include "UavvSetPanTiltVelocity.h"

UavvSetPanTiltVelocity::UavvSetPanTiltVelocity() {}
UavvSetPanTiltVelocity::~UavvSetPanTiltVelocity() {}

UavvSetPanTiltVelocity::UavvSetPanTiltVelocity(float panVelocity, float tiltVelocity)
{
	PanVelocity = panVelocity;
	TiltVelocity = tiltVelocity;
}

ParseResult UavvSetPanTiltVelocity::TryParse(GimbalPacket packet, UavvSetPanTiltVelocity *setPanTiltVelocity)
{
    if (packet.Data.size() < setPanTiltVelocity->Length)
	{
        return ParseResult::InvalidLength;
	}
    float panVelocity;
    float tiltVelocity;
	panVelocity = UavvPacketHelper::PacketToVelocity(packet.Data[0], packet.Data[1]);
	tiltVelocity = UavvPacketHelper::PacketToVelocity(packet.Data[1], packet.Data[2]);
    *setPanTiltVelocity = UavvSetPanTiltVelocity(panVelocity, tiltVelocity);
    return ParseResult::Success;
}

GimbalPacket UavvSetPanTiltVelocity::Encode()
{
	unsigned char data[4];
    vector<unsigned char>tmp_PanVelocity;
    vector<unsigned char>tmp_TiltVelocity;
	tmp_PanVelocity = UavvPacketHelper::VelocityToPacket(PanVelocity);
	for (int i = 0; i < 2; i++)
	{
        data[i] = tmp_PanVelocity[i];
	}

	tmp_TiltVelocity = UavvPacketHelper::VelocityToPacket(TiltVelocity);
	for (int i = 0; i < 2; i++)
	{
        data[2 + i] = tmp_TiltVelocity[i];
	}
	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::SetPanandTilVelocity, data, sizeof(data));
	return result;
}
