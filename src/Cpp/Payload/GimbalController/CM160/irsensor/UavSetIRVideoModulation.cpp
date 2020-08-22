#include"UavSetIRVideoModulation.h"

UavvSetIRVideoModulation::UavvSetIRVideoModulation(){}

UavvSetIRVideoModulation::UavvSetIRVideoModulation(VideoModulation module)
{
    VideoModule = module;
}

UavvSetIRVideoModulation::~UavvSetIRVideoModulation(){}

GimbalPacket UavvSetIRVideoModulation::Encode()
{
	unsigned char data[2];
	data[0] = 0;
    if (VideoModule == VideoModulation::PAL)
		data[1] = 0;
	else
		data[1] = 1;
    return GimbalPacket(UavvGimbalProtocol::SetIRVideoStandard, data, sizeof(data));
}

ParseResult UavvSetIRVideoModulation::TryParse(GimbalPacket packet, UavvSetIRVideoModulation *SetIRVideoModulation)
{
    if (packet.Data.size() < SetIRVideoModulation->Length)
	{
        return ParseResult::InvalidLength;
	}
	if (packet.Data[0] != 0x00)
        return ParseResult::InvalidData;

	VideoModulation _module;
	if (packet.Data[1] == 0x00)
        _module = VideoModulation::PAL;
	else if (packet.Data[1] == 0x01)
        _module = VideoModulation::NTSC;
	else
        return ParseResult::InvalidData;
	
    *SetIRVideoModulation = UavvSetIRVideoModulation(_module);
    return ParseResult::Success;
};
