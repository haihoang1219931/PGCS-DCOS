#include "UavvSaveParameters.h"

UavvSaveParameters::UavvSaveParameters(unsigned char parameter)
{
	Parameter = parameter;
}

UavvSaveParameters::~UavvSaveParameters() {}
UavvSaveParameters::UavvSaveParameters() {}
ParseResult UavvSaveParameters::TryParse(GimbalPacket packet, UavvSaveParameters *parameter)
{
    if (packet.Data.size() < parameter->Length)
	{
        return ParseResult::InvalidLength;
	}

	parameter->Parameter = packet.Data[1];
    return ParseResult::Success;
}

GimbalPacket UavvSaveParameters::Encode()
{
	unsigned char data[2];
	data[0] = 1;
	data[1] = Parameter;
	return GimbalPacket(UavvGimbalProtocol::SaveOffsets, data, sizeof(data));
}
