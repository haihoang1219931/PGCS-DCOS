#include "UavvSetPanVelocity.h"

UavvSetPanVelocity::UavvSetPanVelocity() {}
UavvSetPanVelocity::~UavvSetPanVelocity() {}

UavvSetPanVelocity::UavvSetPanVelocity(float panVelocity)
{
	PanVelocity = panVelocity;
}

ParseResult UavvSetPanVelocity::TryParse(GimbalPacket packet, UavvSetPanVelocity *setPanVelocity)
{
    if (packet.Data.size() < setPanVelocity->Length)
	{
        return ParseResult::InvalidLength;
	}
    float panVelocity;
	panVelocity = UavvPacketHelper::PacketToVelocity(packet.Data[0], packet.Data[1]);
    *setPanVelocity = UavvSetPanVelocity(panVelocity);
    return ParseResult::Success;
}

GimbalPacket UavvSetPanVelocity::Encode()
{
	unsigned char data[2];
    vector<unsigned char>tmp_PanVelocity;

	tmp_PanVelocity = UavvPacketHelper::VelocityToPacket(PanVelocity);

	for (int i = 0; i < 2; i++)
	{
        data[i] = tmp_PanVelocity[i];
	}

	GimbalPacket result = GimbalPacket(UavvGimbalProtocol::SetPanVelocity, data, sizeof(data));
	return result;
}
