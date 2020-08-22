#ifndef UAVENABLEIRISOTHERM_H
#define UAVENABLEIRISOTHERM_H

#include"../UavvPacket.h"

class UavvEnableIRIsotherm
{
public:
    unsigned int Length = 2;
	IsothermStatus Status;

    UavvEnableIRIsotherm();
    UavvEnableIRIsotherm(IsothermStatus status);
    ~UavvEnableIRIsotherm();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableIRIsotherm *EnableIRIsotherm);
};

#endif
