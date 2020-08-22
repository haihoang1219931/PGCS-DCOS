#include "UavvSetPanTiltPosition.h"
UavvSetPanTiltPosition::UavvSetPanTiltPosition() {}
UavvSetPanTiltPosition::~UavvSetPanTiltPosition() {}

UavvSetPanTiltPosition::UavvSetPanTiltPosition(float panPosition, float tiltPosition)
{
	PanPosition = panPosition;
	TiltPosition = tiltPosition;
}

ParseResult UavvSetPanTiltPosition::TryParse(GimbalPacket packet,UavvSetPanTiltPosition *setPanTiltPositionPacket)
{
    if (packet.Data.size() < setPanTiltPositionPacket->Length)
	{
        return ParseResult::InvalidLength;
	}
    float panPosition;
    float tiltPosition;
	panPosition = UavvPacketHelper::PacketToAngle(packet.Data[0], packet.Data[1]);
	tiltPosition = UavvPacketHelper::PacketToAngle(packet.Data[1], packet.Data[2]);
    setPanTiltPositionPacket->PanPosition = panPosition;
    setPanTiltPositionPacket->TiltPosition = tiltPosition;
    return ParseResult::Success;
}

GimbalPacket UavvSetPanTiltPosition::Encode()
{
	unsigned char data[4];
    vector<unsigned char>tmp_PanPosition;
    vector<unsigned char>tmp_TiltPosition;
	tmp_PanPosition = UavvPacketHelper::PositionToPacket(PanPosition);
	tmp_TiltPosition = UavvPacketHelper::PositionToPacket(TiltPosition);
	for (int i = 0; i < 2; i++)
	{
        data[i] = tmp_PanPosition[i];
	}
	for (int i = 0; i < 2; i++)
	{
        data[2 + i] = tmp_TiltPosition[i];
	}
    GimbalPacket result((unsigned char)UavvGimbalProtocol::SetPanandTiltPosition, data,sizeof(data));
    return result;
}

