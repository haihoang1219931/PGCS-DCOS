#ifndef UAVVCURRENTRECORDINGSTATE_H
#define UAVVCURRENTRECORDINGSTATE_H

#include "../UavvPacket.h"

class UavvCurrentRecordingState
{
public:
    unsigned int Length = 9;
	unsigned char Recording;
    unsigned int Reserved01 = 0;
    unsigned int Reserved02 = 0;

	void setRecordingCurrentRecordingState(unsigned char recording)
	{
		Recording = recording;
	}

	
	unsigned char getRecordingCurrentRecordingState()
	{
		return Recording;
	}

	UavvCurrentRecordingState();
	~UavvCurrentRecordingState();
	UavvCurrentRecordingState(unsigned char recording);
    UavvCurrentRecordingState(unsigned char recording,unsigned int reserved01,unsigned int reserved02);
	static ParseResult TryParse(GimbalPacket packet, UavvCurrentRecordingState *CurrentRecordingState);
    GimbalPacket Encode();
};
#endif
