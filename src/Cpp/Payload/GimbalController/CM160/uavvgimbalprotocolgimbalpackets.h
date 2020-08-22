#ifndef UAVVGIMBALPROTOCOLGIMBALPACKETS_H
#define UAVVGIMBALPROTOCOLGIMBALPACKETS_H

/**
*
* Author: hainh35
* Brief : Get and set configuration of ptz system
*
**/
#include <QObject>
#include <QUdpSocket>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "gimbal/UavvCurrentGimbalMode.h"
#include "gimbal/UavvCurrentGimbalPositionRate.h"
#include "gimbal/UavvInitialiseGimbal.h"
#include "gimbal/UavvPanPositionReply.h"
#include "gimbal/UavvSceneSteering.h"
#include "gimbal/UavvSceneSteeringConfiguration.h"
#include "gimbal/UavvSetPanPosition.h"
#include "gimbal/UavvSetPanTiltPosition.h"
#include "gimbal/UavvSetPanTiltVelocity.h"
#include "gimbal/UavvSetPanVelocity.h"
#include "gimbal/UavvSetPrimaryVideo.h"
#include "gimbal/UavvSetTiltPositon.h"
#include "gimbal/UavvSetTiltVelocity.h"
#include "gimbal/UavvStowConfiguration.h"
#include "gimbal/UavvStowMode.h"
#include "gimbal/UavvStowStatusResponse.h"
#include "gimbal/UavvTiltPositionReply.h"

#include "system/UavvRequestResponse.h"
using namespace std;
class UavvGimbalProtocolGimbalPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolGimbalPackets(QObject *parent = 0);
    Q_INVOKABLE void stowConfiguration(
            unsigned char saveToFlash,
            unsigned char enableStow,
            unsigned short stowTimeout,
            unsigned short stowedOnPan,
            unsigned short stowedOnTilt);
    Q_INVOKABLE void stowConfiguration();
    Q_INVOKABLE void stowMode(QString stowMode);
    Q_INVOKABLE void stowStatusResponse();
    Q_INVOKABLE void initiliseGimbal(bool enable);
    Q_INVOKABLE void setPanPosition(float angle);
    Q_INVOKABLE void setTiltPosition(float angle);
    Q_INVOKABLE void setPanTiltPosition(float anglePan, float angleTilt);
    Q_INVOKABLE void setPanVelocity(float velocity);
    Q_INVOKABLE void setTiltVelocity(float velocity);
    Q_INVOKABLE void setPanTiltVelocity(float velPan, float velTilt );
    Q_INVOKABLE void getCurrentGimbalMode();
    Q_INVOKABLE void setPrimaryVideo(int data01,int primaryVideoSensor);
    Q_INVOKABLE void setSceneSteering(bool enable);
    Q_INVOKABLE void getSceneSteering();
    Q_INVOKABLE void setSceneSteeringConfiguration(SceneSteeringAction autoFlags);
    Q_INVOKABLE void getSceneSteeringConfiguration();
};

#endif // UAVVGIMBALPROTOCOLGIMBALPACKETS_H
