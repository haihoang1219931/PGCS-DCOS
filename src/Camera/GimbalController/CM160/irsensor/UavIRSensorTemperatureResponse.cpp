#include"UavIRSensorTemperatureResponse.h"

UavvIRSensorTemperatureResponse::UavvIRSensorTemperatureResponse()
{

}

UavvIRSensorTemperatureResponse::UavvIRSensorTemperatureResponse(float t)
{
    Temperature = t;
}

UavvIRSensorTemperatureResponse::~UavvIRSensorTemperatureResponse(){}

GimbalPacket UavvIRSensorTemperatureResponse::Encode()
{
	unsigned char data[2];

    ByteManipulation::ToBytes((unsigned short)(10 * Temperature),Endianness::Big, data,0);
    return GimbalPacket(UavvGimbalProtocol::IRSensorTemperature, data, sizeof(data));
}

ParseResult UavvIRSensorTemperatureResponse::TryParse(GimbalPacket packet, UavvIRSensorTemperatureResponse *IRSensorTemperatureResponse)
{
    if (packet.Data.size() < IRSensorTemperatureResponse->Length)
	{
        return ParseResult::InvalidLength;
	}

    float t = (float)ByteManipulation::ToUInt16(packet.Data.data(),0,Endianness::Big) / 10;
    *IRSensorTemperatureResponse = UavvIRSensorTemperatureResponse(t);
    return ParseResult::Success;
};
