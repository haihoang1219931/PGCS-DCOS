#include "UavvEnableGyroStabilisation.h"

UavvEnableGyroStabilisation::UavvEnableGyroStabilisation() {}
UavvEnableGyroStabilisation::~UavvEnableGyroStabilisation() {}

UavvEnableGyroStabilisation::UavvEnableGyroStabilisation(unsigned char panFlag,unsigned char tiltFlag)
{
    PanFlag = panFlag;
    TiltFlag = tiltFlag;
}

ParseResult UavvEnableGyroStabilisation::TryParse(GimbalPacket packet, UavvEnableGyroStabilisation *enableGyroGimbalPacket)
{
    if (packet.Data.size() < enableGyroGimbalPacket->Length)
	{
        printf("UavvEnableGyroStabilisation InvalidLength\r\n");
        return ParseResult::InvalidLength;
	}
    enableGyroGimbalPacket->PanFlag = packet.Data[0];
    enableGyroGimbalPacket->TiltFlag = packet.Data[1];
    return ParseResult::Success;
}
GimbalPacket UavvEnableGyroStabilisation::Encode()
{
	unsigned char data[2];
    data[0] = PanFlag;
    data[1] = TiltFlag;
	return GimbalPacket(UavvGimbalProtocol::GyroStablisation, data, sizeof(data));
}
