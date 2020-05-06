#include "UavvVideoDestination.h"


UavvVideoDestination::UavvVideoDestination() {}
UavvVideoDestination::~UavvVideoDestination() {}

UavvVideoDestination::UavvVideoDestination(unsigned int address, unsigned short port, unsigned char networkStream, unsigned char analogOut)
{
	setIPAdressVideoDestination(address);
	setPortVideoDestination(port);
	setNerworkStreamVideoDestination(networkStream);
	setAnalogOutDestination(analogOut);
}

ParseResult UavvVideoDestination::TryParse(GimbalPacket packet, UavvVideoDestination*VideoDestination)
{
    if (packet.Data.size() < VideoDestination->Length)
	{
        return ParseResult::InvalidLength;
	}
    unsigned int address;
    unsigned short port;
	unsigned char networkStream, analogOut;

    address = ByteManipulation::ToUInt32(packet.Data.data(),0,Endianness::Big);
    port = ByteManipulation::ToUInt16(packet.Data.data(),4,Endianness::Big);
	networkStream = packet.Data[6];
	analogOut = packet.Data[7];
    VideoDestination->IPAddress = address;
    VideoDestination->Port = port;
    VideoDestination->NetworkStream = networkStream;
    VideoDestination->AnalogOut = analogOut;
	return ParseResult::Success;
}

GimbalPacket UavvVideoDestination::Encode()
{
	unsigned char data[8];

    ByteManipulation::ToBytes(IPAddress,Endianness::Big,data,0);
    ByteManipulation::ToBytes(Port,Endianness::Big,data,4);
	data[6] = getNetworkStreamVideoDestination();
	data[7] = getAnalogOutVideoDestination();
	return GimbalPacket(UavvGimbalProtocol::SetVideoDestination, data, sizeof(data));
}
