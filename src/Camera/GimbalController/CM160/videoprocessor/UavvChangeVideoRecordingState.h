#ifndef UAVVCHANGEVIDEORECORDINGSTATE_H
#define UAVVCHANGEVIDEORECORDINGSTATE_H

#include "../UavvPacket.h"

class UavvChangeVideoRecordingState
{
public:
    unsigned int Length = 2;
	unsigned char Recording;
    unsigned char Reserved = 0;

	void setRecordingChangeVideoRecordingState(unsigned char recording)
	{
		Recording = recording;
	}

	unsigned char getRecordingChangeVideoRecordingState()
	{
		return Recording;
	}

	UavvChangeVideoRecordingState();
	~UavvChangeVideoRecordingState();
    UavvChangeVideoRecordingState(unsigned char recording,unsigned char reserved);
	static ParseResult TryParse(GimbalPacket packet, UavvChangeVideoRecordingState *ChangeVideoRecordingState);
    GimbalPacket Encode();
};
#endif
