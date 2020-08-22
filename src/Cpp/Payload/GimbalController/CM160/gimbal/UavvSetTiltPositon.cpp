#include "UavvSetTiltPositon.h"

UavvSetTiltPosition::UavvSetTiltPosition() {}
UavvSetTiltPosition::~UavvSetTiltPosition() {}

UavvSetTiltPosition::UavvSetTiltPosition(float tiltPosition)
{
	TiltPosition = tiltPosition;
}

ParseResult UavvSetTiltPosition::TryParse(GimbalPacket packet, UavvSetTiltPosition *settiltPositionPacket)
{
    if (packet.Data.size() < settiltPositionPacket->Length)
	{
        return ParseResult::InvalidLength;
	}
    float tiltPosition;
	tiltPosition = UavvPacketHelper::PacketToAngle(packet.Data[0], packet.Data[1]);
    settiltPositionPacket->TiltPosition = tiltPosition;
    return ParseResult::Success;
}

GimbalPacket UavvSetTiltPosition::Encode()
{
	unsigned char data[2];
    vector<unsigned char>tmp_TiltPosition;
	tmp_TiltPosition = UavvPacketHelper::PositionToPacket(TiltPosition);
	for (int i = 0; i < 2; i++)
	{
        data[i] = tmp_TiltPosition[i];
    }
    return GimbalPacket(UavvGimbalProtocol::SetTiltPosition, data, sizeof(data));
}
