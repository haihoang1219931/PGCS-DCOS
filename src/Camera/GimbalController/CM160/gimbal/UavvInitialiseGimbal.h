#ifndef UAVVINITIALISEGIMBAL_H
#define UAVVINITIALISEGIMBAL_H
#include "../UavvPacket.h"

enum class InitialiseGimbalFlag
{
    DisableGimbal,
    FullInitialisation,
    CalibrateSensors,
    ArmMotors,
    DisamMotors,
};
class UavvInitialiseGimbal
{
public:
    unsigned int Length = 2;
    unsigned char data01;
    unsigned char data02;
    UavvInitialiseGimbal(InitialiseGimbalFlag initialiseGimbalFlag);
	UavvInitialiseGimbal();
	~UavvInitialiseGimbal();
	static ParseResult TryParse(GimbalPacket packet, UavvInitialiseGimbal *initialiseGimbalPacket);
	GimbalPacket Encode();
};
#endif
