#ifndef UAVENABLEAUTOFOCUS_H
#define UAVENABLEAUTOFOCUS_H

#include"../UavvPacket.h"



class UavvEnableAutoFocus
{
public:
    unsigned int Length = 2;
    AutoFocusStatus Status;
    UavvEnableAutoFocus();
    UavvEnableAutoFocus(bool enable);
    UavvEnableAutoFocus(AutoFocusStatus status);
    ~UavvEnableAutoFocus();
	GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvEnableAutoFocus *EnableAutoFocus);
};

#endif
