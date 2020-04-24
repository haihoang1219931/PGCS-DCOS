#ifndef UAVVGIMBALPROTOCOLSYSTEMPACKETS_H
#define UAVVGIMBALPROTOCOLSYSTEMPACKETS_H

/**
*
* Author: hainh35
* Brief : Get and set information of gimbal system
*
**/
#include <QUdpSocket>
#include <QObject>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "system/UavvConfigurePacketRates.h"
#include "system/UavvEnableGyroStabilisation.h"
#include "system/UavvEnableMessageACK.h"
#include "system/UavvEnableStreamMode.h"
#include "system/UavvMessageACKResponse.h"
#include "system/UavvNetworkConfiguration.h"
#include "system/UavvRequestResponse.h"
#include "system/UavvSaveParameters.h"
#include "system/UavvSetSystemTime.h"
#include "system/UavvVersion.h"

using namespace std;
class UavvGimbalProtocolSystemPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolSystemPackets(QObject *parent = 0);
    Q_INVOKABLE void enableGyroStabilisation(bool enable);
    Q_INVOKABLE void getGyroStabilisation();
    Q_INVOKABLE void messageAcknolegdeResponse(int packetID);
    Q_INVOKABLE void enableMessageAcknowledge(int data1, int data2);
    Q_INVOKABLE void enableStreamingMode(EnableStreamTypeActionFlag mode,
                                         EnableStreamFrequencyFlag freq);
    Q_INVOKABLE void getVersion();
    Q_INVOKABLE void setProtocolVersion();
    Q_INVOKABLE void getProtocolVersion();
    Q_INVOKABLE void getGimbalSerialNumberResponse();
    Q_INVOKABLE void requestResponse(int packetID);
    Q_INVOKABLE void configurePacketRates(vector<PacketRate> lstPacketRate);
    Q_INVOKABLE void setSystemTime(int second);
    Q_INVOKABLE void setNetworkConfiguration(
            uint8_t reserved,
            uint8_t ipType,
            uint32_t ipAddress,
            uint32_t subnetMask,
            uint32_t gateWay,
            uint16_t reserved01,
            uint16_t reserved02,
            uint8_t reserved03,
            uint8_t saved);
    Q_INVOKABLE void getNetworkConfiguration();
    Q_INVOKABLE void saveParameters(uint8_t pa01,uint8_t saveParametersFlag);
};

#endif // UAVVGIMBALPROTOCOLSYSTEMPACKETS_H
