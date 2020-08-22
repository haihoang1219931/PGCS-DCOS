#include "UavvStowConfiguration.h"
UavvStowConfiguration::UavvStowConfiguration(unsigned char saveFlash,
                                             unsigned char enableAutoStow,
                                             unsigned short stowedTimeoutPeriod,
                                             unsigned short stowedPan,
                                             unsigned short stowedTilt)
{
    SaveFlash = saveFlash;
    EnableAutoStow = enableAutoStow;
    StowedTimeoutPeriod = stowedTimeoutPeriod;
    StowedPan = stowedPan;
    StowedTilt = stowedTilt;
}

UavvStowConfiguration::~UavvStowConfiguration() {}
UavvStowConfiguration::UavvStowConfiguration() {}
ParseResult UavvStowConfiguration::TryParse(GimbalPacket packet, UavvStowConfiguration *stowConfiguration)
{
    if (packet.Data.size() < stowConfiguration->Length)
	{
        return ParseResult::InvalidLength;
	}
    stowConfiguration->SaveFlash = packet.Data[0];
    stowConfiguration->EnableAutoStow = packet.Data[1];
    stowConfiguration->StowedTimeoutPeriod = ByteManipulation::ToUInt16(packet.Data.data(), 2, Endianness::Big);
    stowConfiguration->StowedPan = ByteManipulation::ToUInt16(packet.Data.data(), 4, Endianness::Big);
    stowConfiguration->StowedTilt = ByteManipulation::ToUInt16(packet.Data.data(), 6, Endianness::Big);
    return ParseResult::Success;
}

GimbalPacket UavvStowConfiguration::encode()
{
	unsigned char data[8];
    data[0] = SaveFlash;
    data[1] = EnableAutoStow;
    data[2] = (StowedTimeoutPeriod >>8) & 0xFF;
    data[3] = (StowedTimeoutPeriod) & 0xFF;
    data[4] = (StowedPan>>8) & 0xFF;
    data[5] = (StowedPan) & 0xFF;
    data[6] = (StowedTilt>>8) & 0xFF;
    data[7] = (StowedTilt) & 0xFF;
	return GimbalPacket(UavvGimbalProtocol::SensorDefog, data, sizeof(data));
}
