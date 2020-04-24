#include "UavMWIRTempPreset.h"

UavvMWIRTempPreset::UavvMWIRTempPreset()
{

}
GimbalPacket UavvMWIRTempPreset::Encode()
{
    unsigned char data[2];
    data[0] = Mode[0];
    data[1] = Mode[1];
    return GimbalPacket(UavvGimbalProtocol::SetMWIRTempPreset, data, sizeof(data));
}

ParseResult UavvMWIRTempPreset::TryParse(GimbalPacket packet, UavvMWIRTempPreset *SetIRPalette)
{
    if (packet.Data.size() < SetIRPalette->Length)
    {
        return ParseResult::InvalidLength;
    }

    unsigned char _mode[2];
    SetIRPalette->Mode[0] = _mode[0];
    SetIRPalette->Mode[1] = _mode[1];
    return ParseResult::Success;
}
