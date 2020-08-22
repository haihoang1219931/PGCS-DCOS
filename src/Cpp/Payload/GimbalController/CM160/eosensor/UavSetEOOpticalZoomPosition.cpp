#include"UavSetEOOpticalZoomPosition.h"

UavvSetEOOpticalZoomPosition::UavvSetEOOpticalZoomPosition(){}

UavvSetEOOpticalZoomPosition::UavvSetEOOpticalZoomPosition(unsigned short position)
{
    Position = position;
}

UavvSetEOOpticalZoomPosition::~UavvSetEOOpticalZoomPosition(){}

GimbalPacket UavvSetEOOpticalZoomPosition::Encode()
{
	unsigned char data[2];
    ByteManipulation::ToBytes(Position,Endianness::Big,data,0);
    return GimbalPacket(UavvGimbalProtocol::SetCameraZoomPosition, data, sizeof(data));
}

ParseResult UavvSetEOOpticalZoomPosition::TryParse(GimbalPacket packet, UavvSetEOOpticalZoomPosition *SetEOOpticalZoomPosition)
{
    if (packet.Data.size() < SetEOOpticalZoomPosition->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned short eoOpticalZoomPosition = ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big);
	*SetEOOpticalZoomPosition = UavvSetEOOpticalZoomPosition(eoOpticalZoomPosition);
    return ParseResult::Success;
}
