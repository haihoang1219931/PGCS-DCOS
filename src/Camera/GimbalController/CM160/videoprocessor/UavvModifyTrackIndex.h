#ifndef UAVVMODIFYTRACKINDEX_H
#define UAVVMODIFYTRACKINDEX_H

#include "../UavvPacket.h"

enum class TrackByIndexAction
{
    StartGimbalTracking = 0x00,
    MakeTrackPrimary = 0x01,
    KillTrack = 0x02
};
class UavvModifyTrackIndex
{
public:
    unsigned int Length = 2;
    TrackByIndexAction Action;
    unsigned char Index;

    void setActModifyTrackIndex(TrackByIndexAction act)
	{
		Action = act;
	}

	void setIndexModifyTrackIndex(unsigned char index)
	{
		Index = index;
	}

    TrackByIndexAction getActModifyTrackIndex()
	{
		return Action;
	}

	unsigned char getIndexModifyObjectTrack()
	{
		return Index;
	}

public:
	UavvModifyTrackIndex();
	~UavvModifyTrackIndex();
    UavvModifyTrackIndex(TrackByIndexAction action, unsigned char index);
	static ParseResult TryParse(GimbalPacket packet, UavvModifyTrackIndex *ModifyTrackIndex);
    GimbalPacket Encode();
};
#endif
