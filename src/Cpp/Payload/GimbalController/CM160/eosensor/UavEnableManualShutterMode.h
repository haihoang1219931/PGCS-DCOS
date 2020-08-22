#ifndef UAVENABLEMANUALSHUTTERMODE_H
#define UAVENABLEMANUALSHUTTERMODE_H

#include"../UavvPacket.h"

class UavvEnableManualShutterMode
{
public:
    unsigned int Length = 2;
	ManualShutterStatus Status;

    UavvEnableManualShutterMode();
    UavvEnableManualShutterMode(bool enable);
    UavvEnableManualShutterMode(ManualShutterStatus status);
    ~UavvEnableManualShutterMode();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableManualShutterMode *EnableManualShutterMode);
};

#endif
