#include "UavvH264StreamParameters.h"


UavvH264StreamParameters::UavvH264StreamParameters() {}
UavvH264StreamParameters::~UavvH264StreamParameters() {}

UavvH264StreamParameters::UavvH264StreamParameters(unsigned int bitRate, unsigned char interval, unsigned char step, unsigned char downSample,unsigned char reserved)
{
    Reserved = reserved;
	setBitRateH264StreamParameters(bitRate);
	setIntervalH264StreamParameters(interval);
	setStepH264StreamParameters(step);
	setDownSampleH264StreamParameters(downSample);
}

ParseResult UavvH264StreamParameters::TryParse(GimbalPacket packet, UavvH264StreamParameters*H264StreamParameters)
{
    if (packet.Data.size() < H264StreamParameters->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned int bitRate;
    unsigned char interval, step, downSample, reserved;

    bitRate = ByteManipulation::ToUInt32(packet.Data.data(),0,Endianness::Big);
    interval = packet.Data[4];
	step = packet.Data[5];
	downSample = packet.Data[6];
    reserved = packet.Data[7];
    *H264StreamParameters = UavvH264StreamParameters(bitRate,interval, step, downSample,reserved);
	return ParseResult::Success;
}

GimbalPacket UavvH264StreamParameters::Encode()
{
	unsigned char data[8];

    ByteManipulation::ToBytes(BitRate,Endianness::Big,data,0);
	data[4] = getIntervalH264StreamParameters();
	data[5] = getStepH264StreamParameters();
	data[6] = getDownSampleH264StreamParameters();
    data[7] = Reserved;
	return GimbalPacket(UavvGimbalProtocol::SetH264Parameters, data, sizeof(data));
}
