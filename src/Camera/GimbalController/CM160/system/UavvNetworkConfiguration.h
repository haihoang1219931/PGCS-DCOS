#ifndef UAVVNETWORKCONFIGURATION_H
#define UAVVNETWORKCONFIGURATION_H

#include "../UavvPacket.h"
class UavvNetworkConfiguration
{
public:
    unsigned int Length = 20;
    unsigned char Reserved1=0;
    unsigned char ModeIP; //Static or dynamic
    unsigned char IPAddr[4];
    unsigned char SubnetMask[4];
    unsigned char GateWay[4];
    unsigned char Reserved2[2];
    unsigned char Reserved3[2];
    unsigned char Reserved4=0;
    unsigned char ConfigurationBootFlag;

    UavvNetworkConfiguration(unsigned char modeip, unsigned char *ipaddr, unsigned char *subnetmask, unsigned char *gateway, unsigned char configurationbootflag);
    ~UavvNetworkConfiguration();
    UavvNetworkConfiguration();
    static ParseResult TryParse(GimbalPacket packet, UavvNetworkConfiguration *networkConfiguration);
    GimbalPacket Encode();
};

#endif // UAVVNETWORKCONFIGURATION_H
