#ifndef UAVVGIMBALPROTOCOLEOSENSORPACKETS_H
#define UAVVGIMBALPROTOCOLEOSENSORPACKETS_H
/**
*
* Author: hainh35
* Brief : Get and set configuration of EO camera
*
**/
#include <QObject>
#include <QUdpSocket>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "eosensor/UavCombinedZoomEnable.h"
#include "eosensor/UavCurrentExposureMode.h"
#include "eosensor/UavDefog.h"
#include "eosensor/UavDisableInfraredCutFilter.h"
#include "eosensor/UavEnableAutoExposure.h"
#include "eosensor/UavEnableAutoFocus.h"
#include "eosensor/UavEnableDigitalZoom.h"
#include "eosensor/UavEnableEOSensor.h"
#include "eosensor/UavEnableLensStabilization.h"
#include "eosensor/UavEnableManualIris.h"
#include "eosensor/UavEnableManualShutterMode.h"
#include "eosensor/UavGetZoomPosition.h"
#include "eosensor/UavInvertPicture.h"
#include "eosensor/UavSensorCurrentFoV.h"
#include "eosensor/UavSensorZoom.h"
#include "eosensor/UavSetCameraGain.h"
#include "eosensor/UavSetDigitalZoomPosition.h"
#include "eosensor/UavSetDigitalZoomVelocity.h"
#include "eosensor/UavSetEOOpticalZoomPosition.h"
#include "eosensor/UavSetEOOpticalZoomVelocity.h"
#include "eosensor/UavSetEOSensorVideoMode.h"
#include "eosensor/UavSetFocus.h"
#include "eosensor/UavSetIris.h"
#include "eosensor/UavSetShutterSpeed.h"
#include "eosensor/UavZoomPositionResponse.h"

#include "system/UavvRequestResponse.h"
using namespace std;
class UavvGimbalProtocolEOSensorPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolEOSensorPackets(QObject *parent = 0);

    Q_INVOKABLE void enableEOSensor(bool enable);
    Q_INVOKABLE void getEnableEOSensor();
    Q_INVOKABLE void enableDigitalZoom(bool enable);
    Q_INVOKABLE void combinedZoomEnable(bool enable);
    Q_INVOKABLE void setDigitalZoomPosition(int position);
    Q_INVOKABLE void setDigitalZoomVelocity(QString status,
                                            int velocity);
    Q_INVOKABLE void setEOOpticalZoomVelocity(QString status,
                                              int velocity);
    Q_INVOKABLE void setEOOpticalZoomPosition(int zoomPosition);
    Q_INVOKABLE void enableAutoFocus(bool enable);
    Q_INVOKABLE void disableInfraredCutFilter(bool disable);
    Q_INVOKABLE void setDefog(QString flag);
    Q_INVOKABLE void getDefog();
    Q_INVOKABLE void setFocus(int focus);
    Q_INVOKABLE void enableManualIris(bool enable);
    Q_INVOKABLE void setIris(int data01, int data02);
    Q_INVOKABLE void enableLensStabilisation(bool enable);
    Q_INVOKABLE void getCurrentExpousureMode();
    Q_INVOKABLE void invertPicture(bool invert);
    Q_INVOKABLE void setShutterSpeed(int data01, int data02);
    Q_INVOKABLE void getShutterSpeed();
    Q_INVOKABLE void setCameraGain(int data01, int data02);
    Q_INVOKABLE void getCameraGain();
    Q_INVOKABLE void setEOSensorVideoMode(int reverse, int HDModeFlag);
    Q_INVOKABLE void enableManualShutterMode(bool enable);
    Q_INVOKABLE void enableAutoExposure(bool enable);
    Q_INVOKABLE void getZoomPosition();
    Q_INVOKABLE void setSensorZoom(int sensorIndex,int zoomFlag,
                       int zoomValue, int reverse);
    Q_INVOKABLE void getSensorsCurrentFOV();
};

#endif // UAVVGIMBALPROTOCOLEOSENSORPACKETS_H
