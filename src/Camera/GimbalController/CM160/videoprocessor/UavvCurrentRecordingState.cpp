#include "UavvCurrentRecordingState.h"

UavvCurrentRecordingState::UavvCurrentRecordingState() {}
UavvCurrentRecordingState::~UavvCurrentRecordingState() {}

UavvCurrentRecordingState::UavvCurrentRecordingState(unsigned char recording)
{
	setRecordingCurrentRecordingState(recording);
}
UavvCurrentRecordingState::UavvCurrentRecordingState(unsigned char recording,
                                                     unsigned int reserved01,
                                                     unsigned int reserved02){
    Recording = recording;
    Reserved01 = reserved01;
    Reserved02 = reserved02;
}

ParseResult UavvCurrentRecordingState::TryParse(GimbalPacket packet, UavvCurrentRecordingState*CurrentRecordingState)
{
    if (packet.Data.size() < CurrentRecordingState->Length)
	{
        return ParseResult::InvalidLength;
	}
    CurrentRecordingState->Recording = packet.Data[0];
    CurrentRecordingState->Reserved01 = ByteManipulation::ToUInt32(packet.Data.data(),1,Endianness::Big);
    CurrentRecordingState->Reserved02 = ByteManipulation::ToUInt32(packet.Data.data(),5,Endianness::Big);
	return ParseResult::Success;
}

GimbalPacket UavvCurrentRecordingState::Encode()
{
	unsigned char data[9];

	data[0] = getRecordingCurrentRecordingState();
    ByteManipulation::ToBytes(Reserved01,Endianness::Big,data,1);
    ByteManipulation::ToBytes(Reserved02,Endianness::Big,data,5);
	return GimbalPacket(UavvGimbalProtocol::SetVideoDestination, data, sizeof(data));
}
