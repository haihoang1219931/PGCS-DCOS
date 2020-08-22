#include "UavvNudgeTrack.h"


UavvNudgeTrack::UavvNudgeTrack() {}
UavvNudgeTrack::~UavvNudgeTrack() {}

UavvNudgeTrack::UavvNudgeTrack(char column, char row)
{
	setColumnPixelOffset(column);
	setRowPixelOffset(row);
}

ParseResult UavvNudgeTrack::TryParse(GimbalPacket packet, UavvNudgeTrack *NudgeTrack)
{
    if (packet.Data.size() < NudgeTrack->Length)
	{
        return ParseResult::InvalidLength;
	}
	char column, row;

	column = packet.Data[1];
	row = packet.Data[2];
    *NudgeTrack = UavvNudgeTrack(column, row);
	return ParseResult::Success;
}

GimbalPacket UavvNudgeTrack::Encode()
{
	unsigned char data[3];
	data[0] = 0;
	data[1] = getColumnPixelOffset();
	data[2] = getRowPixelOffset();
	return GimbalPacket(UavvGimbalProtocol::NudgeTrack, data, sizeof(data));
}
