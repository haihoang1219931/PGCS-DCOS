#include "UavvStabiliseOnTrack.h"


UavvStabiliseOnTrack::UavvStabiliseOnTrack() {}
UavvStabiliseOnTrack::~UavvStabiliseOnTrack() {}

UavvStabiliseOnTrack::UavvStabiliseOnTrack(bool enable)
{
	setEnableStabiliseOnTrack(enable);
}

ParseResult UavvStabiliseOnTrack::TryParse(GimbalPacket packet, UavvStabiliseOnTrack *StabiliseOnTrack)
{
    if (packet.Data.size() < StabiliseOnTrack->Length)
	{
        return ParseResult::InvalidLength;
	}
    bool enable;
    enable = packet.Data[0]==0x01?true:false;
    StabiliseOnTrack->Enable = enable;
	return ParseResult::Success;
}

GimbalPacket UavvStabiliseOnTrack::Encode()
{
    printf("Set stabilise on Track = %s\r\n",Enable == true?"true":"false");
	unsigned char data[1];
    data[0] = Enable == true?0x01:0x00;
    return GimbalPacket(UavvGimbalProtocol::StabililseOnTrack, data, sizeof(data));
}
