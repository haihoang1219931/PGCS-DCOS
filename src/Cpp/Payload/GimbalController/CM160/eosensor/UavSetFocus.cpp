#include"UavSetFocus.h"

UavvSetFocus::UavvSetFocus(){}

UavvSetFocus::UavvSetFocus(unsigned short focusPosition)
{
    FocusPosition = focusPosition;
};

UavvSetFocus::~UavvSetFocus(){}

GimbalPacket UavvSetFocus::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes(FocusPosition,Endianness::Big,data,0);
    return GimbalPacket(UavvGimbalProtocol::SetFocus, data, sizeof(data));
}

ParseResult UavvSetFocus::TryParse(GimbalPacket packet, UavvSetFocus *SetFocus)
{
    if (packet.Data.size() < SetFocus->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned short _focus = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
	if ((_focus == 0xFFFF) || (_focus == 0xFFFE) || (_focus == 0xFFFD) || (_focus == 0xFFFB) || ((_focus >= 0x1000) && (_focus <= 0xC000)))
	{
        *SetFocus = UavvSetFocus(_focus);
        return ParseResult::Success;
	}
    return ParseResult::InvalidData;
}
