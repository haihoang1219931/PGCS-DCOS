#ifndef UAVVGIMBALPROTOCOLIRSENSORPACKETS_H
#define UAVVGIMBALPROTOCOLIRSENSORPACKETS_H

/**
*
* Author: hainh35
* Brief : get and set configuration of IR camera
*
**/
#include <QUdpSocket>

#include "irsensor/UavEnableIRIsotherm.h"
#include "irsensor/UavIRSensorTemperatureResponse.h"
#include "irsensor/UavPerformFFC.h"
#include "irsensor/UavResetIRCamera.h"
#include "irsensor/UavSetDynamicDDE.h"
#include "irsensor/UavSetFFCMode.h"
#include "irsensor/UavSetFFCTemperatureDelta.h"
#include "irsensor/UavSetIRAGCMode.h"
#include "irsensor/UavSetIRBrightness.h"
#include "irsensor/UavSetIRBrightnessBias.h"
#include "irsensor/UavSetIRContrast.h"
#include "irsensor/UavSetIRGainMode.h"
#include "irsensor/UavSetIRIsothermThresholds.h"
#include "irsensor/UavSetIRITTMidpoint.h"
#include "irsensor/UavSetIRMaxGain.h"
#include "irsensor/UavSetIRPalette.h"
#include "irsensor/UavSetIRPlateauLevel.h"
#include "irsensor/UavSetIRVideoModulation.h"
#include "irsensor/UavSetIRVideoOrientation.h"
#include "irsensor/UavSetIRZoom.h"
#include "irsensor/UavMWIRTempPreset.h"

#include "system/UavvRequestResponse.h"

class UavvGimbalProtocolIRSensorPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolIRSensorPackets(QObject *parent = 0);
    Q_INVOKABLE void setIRSensorTempResponse();
    Q_INVOKABLE void getIRSensorTempResponse();
    Q_INVOKABLE void setIRVideoModulation(unsigned char reverse,VideoModulation moduleFlag);
    Q_INVOKABLE void getIRVideoModulation();
    Q_INVOKABLE void setIRZoom(int zoomFlag);
    Q_INVOKABLE void setIRAGC(AGCMode agcMode);
    Q_INVOKABLE void setIRBrightness(unsigned short brightness);
    Q_INVOKABLE void getIRBrightness();
    Q_INVOKABLE void enableIRIsotherm(bool enable);
    Q_INVOKABLE void enableIRIsotherm();
    Q_INVOKABLE void resetIRCamera(unsigned char data01,unsigned char data02);
    Q_INVOKABLE void setFFCMode(bool ffcMode);
    Q_INVOKABLE void performFlatFieldCorrection(unsigned char reserve,unsigned char FFC);
    Q_INVOKABLE void setFFCTempDelta(unsigned short temp);
    Q_INVOKABLE void setMWIRTempPreset(QString preset);
    Q_INVOKABLE void setIRPalette(QString palette);
    Q_INVOKABLE void getIRPalette();
    Q_INVOKABLE void setIRVideoOrientation(unsigned char reserve,VideoOrientationMode orienOption);
    Q_INVOKABLE void getIRVideoOrientation();
    Q_INVOKABLE void setIRContrast(unsigned char data01, unsigned char value);
    Q_INVOKABLE void getIRContrast();
    Q_INVOKABLE void setIRBrightnessBias(short value);
    Q_INVOKABLE void getIRBrightnessBias();
    Q_INVOKABLE void setIRPlateanLevel(unsigned short value);
    Q_INVOKABLE void getIRPlateanLevel();
    Q_INVOKABLE void setIRImageTransformTableMidpoint(
            unsigned char data01,unsigned char value);
    Q_INVOKABLE void getIRImageTransformTableMidpoint();
    Q_INVOKABLE void setIRMaxGain(unsigned short value);
    Q_INVOKABLE void getIRMaxGain();
    Q_INVOKABLE void setIRGainMode(unsigned char data01,GainMode mode);
    Q_INVOKABLE void getIRGainMode();
    Q_INVOKABLE void setIRDDE(ManualDDEStatus DDE,unsigned char sharpness);
    Q_INVOKABLE void getIRDDE();
};

#endif // UAVVGIMBALPROTOCOLIRSENSORPACKETS_H
