#include "UavvTakeSnapshot.h"


UavvTakeSnapshot::UavvTakeSnapshot() {}
UavvTakeSnapshot::~UavvTakeSnapshot() {}

UavvTakeSnapshot::UavvTakeSnapshot(unsigned char set)
{
	setSetTakeSnapshot(set);
}

ParseResult UavvTakeSnapshot::TryParse(GimbalPacket packet, UavvTakeSnapshot*TakeSnapshot)
{
    if (packet.Data.size() < TakeSnapshot->Length)
	{
        return ParseResult::InvalidLength;
	}

    TakeSnapshot->Set = packet.Data[0];
    TakeSnapshot->Reserved = ByteManipulation::ToUInt16(packet.Data.data(),1,Endianness::Big);
	return ParseResult::Success;
}

GimbalPacket UavvTakeSnapshot::Encode()
{
	unsigned char data[3];

	data[0] = getSetTakeSnapshot();
    ByteManipulation::ToBytes(Reserved,Endianness::Big,data,1);
	return GimbalPacket(UavvGimbalProtocol::TakeSnapshot, data, sizeof(data));
}
