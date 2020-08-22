#include "UavvCurrentImageSize.h"


UavvCurrentImageSize::UavvCurrentImageSize() {}
UavvCurrentImageSize::~UavvCurrentImageSize() {}

UavvCurrentImageSize::UavvCurrentImageSize(unsigned char reserved,unsigned int frameWidth, unsigned int frameHeight)
{
    Reserved = reserved;
	setFrameWidthCurrentImageSize(frameWidth);
	setFrameHeightCurrentImageSize(frameHeight);
}

ParseResult UavvCurrentImageSize::TryParse(GimbalPacket packet, UavvCurrentImageSize *CurrentImageSize)
{
    if (packet.Data.size() < CurrentImageSize->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned int reserved,frameWidth, frameHeight;
    reserved = packet.Data[0];
    frameWidth = ByteManipulation::ToUInt16(packet.Data.data(),1,Endianness::Big);
    frameHeight = ByteManipulation::ToUInt16(packet.Data.data(),3,Endianness::Big);
    //frameWidth = (packet.Data[1] << 8) | (packet.Data[2]);
    //frameHeight = (packet.Data[3] << 8) | (packet.Data[4]);
    *CurrentImageSize = UavvCurrentImageSize(reserved,frameWidth, frameHeight);
	return ParseResult::Success;
}

GimbalPacket UavvCurrentImageSize::Encode()
{
	unsigned char data[5];
    data[0] = Reserved;
    ByteManipulation::ToBytes(FrameWidth,Endianness::Big,data,1);
    ByteManipulation::ToBytes(FrameHeight,Endianness::Big,data,3);
	return GimbalPacket(UavvGimbalProtocol::ImageSize, data, sizeof(data));
}
