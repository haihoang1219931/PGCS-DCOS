#include "UavvInitialiseGimbal.h"
#include "../UavvGimbalProtocol.h"

UavvInitialiseGimbal::UavvInitialiseGimbal() {}
UavvInitialiseGimbal::~UavvInitialiseGimbal() {}

UavvInitialiseGimbal::UavvInitialiseGimbal(InitialiseGimbalFlag initialiseGimbalFlag)
{
    switch (initialiseGimbalFlag) {
    case InitialiseGimbalFlag::FullInitialisation:
        data01 = 0x01;
        data02 = 0x01;
        break;
    case InitialiseGimbalFlag::DisableGimbal:
        data01 = 0x00;
        data02 = 0x00;
        break;
    case InitialiseGimbalFlag::DisamMotors:
        break;
    case InitialiseGimbalFlag::CalibrateSensors:
        break;
    case InitialiseGimbalFlag::ArmMotors:
        break;
    }
}

ParseResult UavvInitialiseGimbal::TryParse(GimbalPacket packet, UavvInitialiseGimbal *initialiseGimbalPacket)
{
    if (packet.Data.size() < initialiseGimbalPacket->Length)
	{
        return ParseResult::InvalidLength;
	}
    initialiseGimbalPacket->data01 = packet.Data[0];
    initialiseGimbalPacket->data02 = packet.Data[1];
    //*initialiseGimbalPacket = UavvInitialiseGimbal(gimbalInitialiseStatus);
    return ParseResult::Success;
}

GimbalPacket UavvInitialiseGimbal::Encode()
{
	unsigned char data[2];
    data[0] = data01;
    data[1] = data02;
    return GimbalPacket(UavvGimbalProtocol::InitialiseGimbal, data,sizeof(data));
}
