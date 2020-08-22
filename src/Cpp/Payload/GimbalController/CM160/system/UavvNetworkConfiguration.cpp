#include "UavvNetworkConfiguration.h"

UavvNetworkConfiguration::~UavvNetworkConfiguration() {}
UavvNetworkConfiguration::UavvNetworkConfiguration() {}
UavvNetworkConfiguration::UavvNetworkConfiguration(unsigned char modeip, unsigned char *ipaddr, unsigned char *subnetmask, unsigned char *gateway, unsigned char configurationbootflag)
{
    ModeIP = modeip;
    IPAddr[0] = ipaddr[0];
    IPAddr[1] = ipaddr[1];
    IPAddr[2] = ipaddr[2];
    IPAddr[3] = ipaddr[3];
    SubnetMask[0] = subnetmask[0];
    SubnetMask[1] = subnetmask[1];
    SubnetMask[2] = subnetmask[2];
    SubnetMask[3] = subnetmask[3];
    GateWay[0] = gateway[0];
    GateWay[1] = gateway[1];
    GateWay[2] = gateway[2];
    GateWay[3] = gateway[3];
    ConfigurationBootFlag = configurationbootflag;
}

ParseResult UavvNetworkConfiguration::TryParse(GimbalPacket packet, UavvNetworkConfiguration *networkConfiguration)
{
    if (packet.Data.size() < networkConfiguration->Length)
    {
        return ParseResult::InvalidLength;
    }
    networkConfiguration->ModeIP = packet.Data[1];
    for (int i = 0; i < 4; i++)
    {
        networkConfiguration->IPAddr[i] = packet.Data[2 + i];
    }
    for (int i = 0; i < 4; i++)
    {
        networkConfiguration->SubnetMask[i] = packet.Data[6 + i];
    }
    for (int i = 0; i < 4; i++)
    {
        networkConfiguration->GateWay[i] = packet.Data[10 + i];
    }
    networkConfiguration->ConfigurationBootFlag = packet.Data[19];
    return ParseResult::Success;
}

GimbalPacket UavvNetworkConfiguration::Encode()
{
    unsigned char data[20];
    data[0] = 0;
    data[1] = ModeIP;
    for (int i = 0; i < 4; i++)
    {
        data[2 + i] = IPAddr[i];
    }

    for (int i = 0; i < 4; i++)
    {
        data[6 + i] = SubnetMask[i];
    }

    for (int i = 0; i < 4; i++)
    {
        data[10 + i] = GateWay[i];
    }
    data[14] = 0;
    data[15] = 0;
    data[16] = 0;
    data[17] = 0;
    data[18] = 0;
    data[19] = ConfigurationBootFlag;
    return GimbalPacket(UavvGimbalProtocol::NetworkConfiguration, data, sizeof(data));
}
