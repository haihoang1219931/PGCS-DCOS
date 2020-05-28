#include "UavvConfigurePacketRates.h"

PacketRate::PacketRate() {}
PacketRate::~PacketRate() {}

PacketRate::PacketRate(UavvGimbalProtocol packetId, unsigned int period)
{
	PacketId = packetId;
	Period = period;
}

UavvConfigurePacketRates::UavvConfigurePacketRates() {}
UavvConfigurePacketRates::~UavvConfigurePacketRates() {}

UavvConfigurePacketRates::UavvConfigurePacketRates(bool clearExisting, vector<PacketRate> packetRates)
{
	ClearExisting = clearExisting;
	PacketRates = packetRates;
    Length = 1+5*PacketRates.size();
}

ParseResult UavvConfigurePacketRates::TryParse(GimbalPacket packet,UavvConfigurePacketRates *configurePacket)
{
    packet;
    configurePacket;
    return ParseResult::Success;
}

GimbalPacket UavvConfigurePacketRates::Encode()
{
	std::vector<unsigned char> encoded;
	//PacketRate rate;
	encoded.push_back(ClearExisting);
	//for (int i =0; i<PacketRates.size();i++)
    for(PacketRate rate:PacketRates)
	{
		//rate.PacketId = PacketRates[i].PacketId;
		//rate.Period = PacketRates[i].Period;
		encoded.push_back((unsigned char)rate.PacketId);
        encoded.push_back(((unsigned char)rate.Period >> 24) & 0xFF);
        encoded.push_back(((unsigned char)rate.Period >> 16) & 0xFF);
        encoded.push_back(((unsigned char)rate.Period >> 8) & 0xFF);
        encoded.push_back(((unsigned char)rate.Period) & 0xFF);
    }
    return GimbalPacket(UavvGimbalProtocol::ConfigurePacketRates, encoded);
}
