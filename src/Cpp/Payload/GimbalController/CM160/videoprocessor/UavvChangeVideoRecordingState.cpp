#include "UavvChangeVideoRecordingState.h"


UavvChangeVideoRecordingState::UavvChangeVideoRecordingState() {}
UavvChangeVideoRecordingState::~UavvChangeVideoRecordingState() {}

UavvChangeVideoRecordingState::UavvChangeVideoRecordingState(unsigned char recording,unsigned char reserved)
{
	setRecordingChangeVideoRecordingState(recording);
    Reserved = reserved;
}

ParseResult UavvChangeVideoRecordingState::TryParse(GimbalPacket packet, UavvChangeVideoRecordingState*ChangeVideoRecordingState)
{
    if (packet.Data.size() < ChangeVideoRecordingState->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned char recording,reserved;
	recording = packet.Data[0];
    reserved = packet.Data[1];
    *ChangeVideoRecordingState = UavvChangeVideoRecordingState(recording,reserved);
	return ParseResult::Success;
}

GimbalPacket UavvChangeVideoRecordingState::Encode()
{
	unsigned char data[2];

	data[0] = Recording;
    data[1] = Reserved;
	return GimbalPacket(UavvGimbalProtocol::ChangeRecordingStatus, data, sizeof(data));
}
