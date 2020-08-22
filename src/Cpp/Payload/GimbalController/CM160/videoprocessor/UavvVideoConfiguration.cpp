#include "UavvVideoConfiguration.h"


UavvVideoConfiguration::UavvVideoConfiguration() {}
UavvVideoConfiguration::~UavvVideoConfiguration() {}

UavvVideoConfiguration::UavvVideoConfiguration(VideoConfigurationEncoderType encoder,
                                               VideoConfigurationOutputFrameSize sensor0,
                                               VideoConfigurationOutputFrameSize sensor1)
{
	setEncoderVideoConfiguration(encoder);
	setSensor0VideoConfiguration(sensor0);
	setSensor1VideoConfiguration(sensor1);
}

ParseResult UavvVideoConfiguration::TryParse(GimbalPacket packet, UavvVideoConfiguration *VideoConfiguration)
{
    if (packet.Data.size() < VideoConfiguration->Length)
	{
        return ParseResult::InvalidLength;
	}
    VideoConfigurationEncoderType encoder;
    VideoConfigurationOutputFrameSize sensor0;
    VideoConfigurationOutputFrameSize sensor1;
    unsigned short reserved;

	if (packet.Data[0] == 0x00)
        encoder = VideoConfigurationEncoderType::H264Legacy;
	else if (packet.Data[0] == 0x01)
        encoder = VideoConfigurationEncoderType::H264;
	else if (packet.Data[0] == 0x02)
        encoder = VideoConfigurationEncoderType::Mpeg4;
	if (packet.Data[1] == 0x00)
        sensor0 = VideoConfigurationOutputFrameSize::SD;
	else if (packet.Data[1] == 0x01)
        sensor0 = VideoConfigurationOutputFrameSize::Size960x720;
	else if (packet.Data[1] == 0x02)
        sensor0 = VideoConfigurationOutputFrameSize::Size720;
	if (packet.Data[2] == 0x00)
        sensor1 = VideoConfigurationOutputFrameSize::SD;
	else if (packet.Data[2] == 0x01)
        sensor1 = VideoConfigurationOutputFrameSize::Size960x720;
	else if (packet.Data[2] == 0x02)
        sensor1 = VideoConfigurationOutputFrameSize::Size720;
    reserved = ByteManipulation::ToUInt16(packet.Data.data(),3,Endianness::Big);
    VideoConfiguration->EncoderType = encoder;
    VideoConfiguration->Sensor0 = sensor0;
    VideoConfiguration->Sensor1 = sensor1;
    VideoConfiguration->Reseved = reserved;
	return ParseResult::Success;
}

GimbalPacket UavvVideoConfiguration::Encode()
{
	unsigned char data[5];

    data[0] = (unsigned char)getEncoderVideoConfiguration();
    data[1] = (unsigned char)getSensor0VideoConfiguration();
    data[2] = (unsigned char)getSensor1VideoConfiguration();
    ByteManipulation::ToBytes(Reseved,Endianness::Big,data,3);
	return GimbalPacket(UavvGimbalProtocol::VideoConfiguration, data, sizeof(data));
}
