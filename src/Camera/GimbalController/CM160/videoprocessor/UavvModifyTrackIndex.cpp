#include "UavvModifyTrackIndex.h"


UavvModifyTrackIndex::UavvModifyTrackIndex() {}
UavvModifyTrackIndex::~UavvModifyTrackIndex() {}

UavvModifyTrackIndex::UavvModifyTrackIndex(TrackByIndexAction action, unsigned char index)
{
	setActModifyTrackIndex(action);
	setIndexModifyTrackIndex(index);
}

ParseResult UavvModifyTrackIndex::TryParse(GimbalPacket packet, UavvModifyTrackIndex *ModifyTrackIndex)
{
    if (packet.Data.size() < ModifyTrackIndex->Length)
	{
        return ParseResult::InvalidLength;
	}
    TrackByIndexAction action;
    unsigned char index;
	if (packet.Data[0] == 0x00)
        action = TrackByIndexAction::StartGimbalTracking;
	else if (packet.Data[0] == 0x01)
        action = TrackByIndexAction::MakeTrackPrimary;
	else if (packet.Data[0] == 0x02)
        action = TrackByIndexAction::KillTrack;
	else
        return ParseResult::InvalidData;
	index = packet.Data[1];
    ModifyTrackIndex->Action = action;
    ModifyTrackIndex->Index = index;
	return ParseResult::Success;
}

GimbalPacket UavvModifyTrackIndex::Encode()
{
	unsigned char data[2];
    if (getActModifyTrackIndex() == TrackByIndexAction::StartGimbalTracking)
		data[0] = 0x00;
    else if (getActModifyTrackIndex() == TrackByIndexAction::MakeTrackPrimary)
		data[0] = 0x01;
    else if (getActModifyTrackIndex() == TrackByIndexAction::KillTrack)
		data[0] = 0x02;
	data[1] = getIndexModifyObjectTrack();
	return GimbalPacket(UavvGimbalProtocol::ModifyTrackByIndex, data, sizeof(data));
}
