#ifndef UAVVGIMBALPROTOCOLGEOPOINTINGPACKETS_H
#define UAVVGIMBALPROTOCOLGEOPOINTINGPACKETS_H

/**
*
* Author: hainh35
* Brief : Get and set position of the targets on the WGS84 Ellipsoid
*
**/
#include <QObject>
#include <QUdpSocket>
#include <stdio.h>

#include "geopointing/UavvAltitudeOffset.h"
#include "geopointing/UavvCurrentCornerLocations.h"
#include "geopointing/UavvCurrentGeolockSetpoit.h"
#include "geopointing/UavvCurrentTargetLocation.h"
#include "geopointing/UavvGimbalMisalignmentOffset.h"
#include "geopointing/UavvGimbalOrientationOffset.h"
#include "geopointing/UavvGNSSStatus.h"
#include "geopointing/UavvPlatformOrientation.h"
#include "geopointing/UavvPlatformPosition.h"
#include "geopointing/UavvSeedTerrainHeight.h"
#include "geopointing/UavvSetGeolockLocation.h"
#include "geopointing/UavvExternalAltitude.h"
#include "geopointing/UavvExternalPosition.h"

#include "system/UavvRequestResponse.h"
class UavvGimbalProtocolGeoPointingPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolGeoPointingPackets(QObject *parent = 0);
    Q_INVOKABLE void setGeolockLocation(QString actionFlag,
                                        float latitude,
                                        float longtitude,
                                        float height);
    Q_INVOKABLE void currentGeolockSetpoint();
    Q_INVOKABLE void currentTargetLocation();
    Q_INVOKABLE void currentCornerLocation();
    Q_INVOKABLE void terrainHeight();
    Q_INVOKABLE void seedTerrainHeight(float latitude,
                                       float longtitude,
                                       float height);
    Q_INVOKABLE void seedTerrainHeight();
    Q_INVOKABLE void gnssStatus();
    Q_INVOKABLE void platformOrientation();
    Q_INVOKABLE void platformPosition();
    Q_INVOKABLE void gimbalOrientationOffset(int reserve,
                                             int rollOffset,
                                             int pitchOffset,
                                             int yawOffset);
    Q_INVOKABLE void gimbalOrientationOffset();
    Q_INVOKABLE void altitudeOffset(unsigned char reserve,
                                    short altitude);
    Q_INVOKABLE void altitudeOffset();
    Q_INVOKABLE void gimbalMisalignmentOffset(QString mountType,
                                              float panMisalign,
                                              float tiltMisalign);
    Q_INVOKABLE void gimbalMisalignmentOffset();
    Q_INVOKABLE void sendExternalAltitude(float roll,float pitch, float yaw);
    Q_INVOKABLE void sendExternalPosition(float lat,float lon, float alt);
    Q_INVOKABLE void sendExternalElevation(float lat,float lon, float alt);

};

#endif // UAVVGIMBALPROTOCOLGEOPOINTINGPACKETS_H
