#ifndef UAVVGIMBALPROTOCOLLASERRANGEFINDERPACKETS_H
#define UAVVGIMBALPROTOCOLLASERRANGEFINDERPACKETS_H

/**
*
* Author: hainh35
* Brief : Get position of the gimbal on the WGS84 Ellipsoid
*
**/
#include <QUdpSocket>

#include "laserrangefinder/LaserRange.h"
#include "laserrangefinder/ArmLaserDevice.h"
#include "laserrangefinder/FireLaserDevice.h"
#include "laserrangefinder/LaserDeviceStatus.h"
#include "laserrangefinder/LaserRangeStart.h"
#include "laserrangefinder/LaserRangeStatus.h"
#include "system/UavvRequestResponse.h"

class UavvGimbalProtocolLaserRangeFinderPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolLaserRangeFinderPackets(QObject *parent = 0);
    Q_INVOKABLE void getDistance();
    Q_INVOKABLE void laserRangeStart(bool start);
    Q_INVOKABLE void getLaserDeviceStatus();
    Q_INVOKABLE void armLaserDevice(bool arm);
    Q_INVOKABLE void fireLaserDevice(bool fire);
};

#endif // UAVVGIMBALPROTOCOLLASERRANGEFINDERPACKETS_H
