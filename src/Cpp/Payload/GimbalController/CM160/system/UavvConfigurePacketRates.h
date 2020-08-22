#ifndef UAVVCONFIGUREPACKETRATES_H
#define UAVVCONFIGUREPACKETRATES_H

#include "../UavvPacket.h"
#include <vector>
using namespace std;

class PacketRate
{
public:
    unsigned int Period;
    UavvGimbalProtocol PacketId;
	PacketRate();
	~PacketRate();
    PacketRate(UavvGimbalProtocol packetId, unsigned int period);
};

class UavvConfigurePacketRates
{
public:
	std::vector<PacketRate> PacketRates;
	unsigned short int Period;
	UavvGimbalProtocol PacketID;
	int Length = 1;
	bool ClearExisting;
    UavvConfigurePacketRates();
    ~UavvConfigurePacketRates();
    UavvConfigurePacketRates(bool clearExisting, vector<PacketRate> packetRates);
    static ParseResult TryParse(GimbalPacket packet, UavvConfigurePacketRates *configurePacket);
	GimbalPacket Encode();
};
#endif
