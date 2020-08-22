#include "UavvSetPanPosition.h"

UavvSetPanPosition::UavvSetPanPosition() {}
UavvSetPanPosition::~UavvSetPanPosition() {}

UavvSetPanPosition::UavvSetPanPosition(float panPosition)
{
	PanPosition = panPosition;
}

ParseResult UavvSetPanPosition::TryParse(GimbalPacket packet, UavvSetPanPosition *setPanPositionPacket)
{
    if (packet.Data.size() < setPanPositionPacket->Length)
	{
        return ParseResult::InvalidLength;
	}
    float panPosition;
	panPosition = UavvPacketHelper::PacketToAngle(packet.Data[0], packet.Data[1]);
    setPanPositionPacket->PanPosition = panPosition;
    return ParseResult::Success;
}

GimbalPacket UavvSetPanPosition::Encode()
{
	unsigned char data[2];
    vector<unsigned char>tmp_PanPosition;
	tmp_PanPosition = UavvPacketHelper::PositionToPacket(PanPosition);

	for (int i = 0; i < 2; i++)
	{
        data[i] = tmp_PanPosition[i];
	}

    return GimbalPacket(UavvGimbalProtocol::SetPanPosition, data, sizeof(data));
}
