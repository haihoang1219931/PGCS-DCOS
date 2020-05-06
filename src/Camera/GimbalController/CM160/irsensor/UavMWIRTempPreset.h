#ifndef UAVMWIRTEMPPRESET_H
#define UAVMWIRTEMPPRESET_H

#include "../UavvPacket.h"
class UavvMWIRTempPreset
{
public:
    unsigned int Length = 2;
    unsigned char Reserved = 0;
    unsigned char Mode[2];
    UavvMWIRTempPreset();
    GimbalPacket Encode();
    static ParseResult TryParse(GimbalPacket packet, UavvMWIRTempPreset *SetIRPalette);
};

#endif // UAVMWIRTEMPPRESET_H
