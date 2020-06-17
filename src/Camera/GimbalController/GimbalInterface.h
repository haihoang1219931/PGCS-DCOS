#ifndef GIMBALINTERFACE_H
#define GIMBALINTERFACE_H

#include <QObject>
#include <QString>
#include <QPoint>
#include "GimbalData.h"
#include "Setting/config.h"
#include "../../Controller/Vehicle/Vehicle.h"
#include "Camera/TargetLocation/TargetLocalization.h"
class VideoEngine;
class JoystickThreaded;
class GimbalInterface : public QObject
{
    Q_OBJECT
    Q_PROPERTY(JoystickThreaded*    joystick                    READ joystick       WRITE setJoystick)
    Q_PROPERTY(float digitalZoomMax READ digitalZoomMax NOTIFY digitalZoomMaxChanged)
    Q_PROPERTY(float zoomMax READ zoomMax NOTIFY zoomMaxChanged)
    Q_PROPERTY(float zoomMin READ zoomMin NOTIFY zoomMinChanged)
    Q_PROPERTY(float zoom READ zoom NOTIFY zoomChanged)
    Q_PROPERTY(float zoomTarget READ zoomTarget NOTIFY zoomTargetChanged)
public:
    explicit GimbalInterface(QObject *parent = nullptr);
    GimbalData* context(){ return m_context; }
    void setVideoEngine(VideoEngine* videoEngine);

    virtual void setVehicle(Vehicle* vehicle);

    JoystickThreaded* joystick();
    virtual void setJoystick(JoystickThreaded* joystick);
    float digitalZoomMax(){
        return m_context->m_digitalZoomMax[m_context->m_sensorID];
    }
    void setDigitalZoomMax(int sensorID, float digitalZoomMax){
        if(sensorID >=0 && sensorID < MAX_SENSOR){
            m_context->m_digitalZoomMax[sensorID] = digitalZoomMax;
            Q_EMIT digitalZoomMaxChanged();
        }
    }
    float zoomMax(){
        return m_context->m_zoomMax[m_context->m_sensorID];
    }
    void setZoomMax(int sensorID, float zoomMax){
        if(sensorID >=0 && sensorID < MAX_SENSOR){
            m_context->m_zoomMax[sensorID] = zoomMax;
            Q_EMIT zoomMaxChanged();
        }
    }
    float zoomMin(){
        return m_context->m_zoomMin[m_context->m_sensorID];
    }
    void setZoomMin(int sensorID, float zoomMin){
        if(sensorID >=0 && sensorID < MAX_SENSOR){
            m_context->m_zoomMin[sensorID] = zoomMin;
            Q_EMIT zoomMinChanged();
        }
    }
    float zoom(){
        return m_context->m_zoom[m_context->m_sensorID];
    }
    void setZoom(int sensorID, float zoom){
        if(sensorID >=0 && sensorID < MAX_SENSOR){
            m_context->m_zoom[sensorID] = zoom;
            Q_EMIT zoomChanged();
        }
    }
    float zoomTarget(){
        return m_context->m_zoomTarget[m_context->m_sensorID];
    }
    void setZoomTarget(int sensorID, float zoomTarget){
        if(sensorID >=0 && sensorID < MAX_SENSOR){
            m_context->m_zoomTarget[sensorID] = zoomTarget;
            Q_EMIT zoomTargetChanged();
        }
    }
    float zoomCalculated(){
        return m_context->m_zoomCalculated[m_context->m_sensorID];
    }
    void setZoomCalculated(int sensorID, float zoomCalculated){
        if(sensorID >=0 && sensorID < MAX_SENSOR){
            m_context->m_zoomCalculated[sensorID] = zoomCalculated;
            Q_EMIT zoomCalculatedChanged(sensorID,zoomCalculated);
        }
    }
Q_SIGNALS:
    void digitalZoomMaxChanged();
    void zoomMaxChanged();
    void zoomMinChanged();
    void zoomChanged();
    void zoomTargetChanged();
    void zoomCalculatedChanged(int viewIndex,float zoomCalculated);
    void functionHandled(QString message);
    void presetChanged(bool result);
public Q_SLOTS:
    virtual void connectToGimbal(Config* config = nullptr);
    virtual void disconnectGimbal();
    virtual void discoverOnLan();
    virtual void changeSensor(QString sensorID);
    virtual void setSensorColor(QString sensorID,QString colorMode);
    virtual void handleAxes();
    virtual void lockScreenPoint(int _id,double _px,double _py,double _oW,double _oH,double _w,double _h);
    virtual void setPanRate(float rate);
    virtual void setTiltRate(float rate);
    virtual void setGimbalRate(float panRate,float tiltRate);
    virtual void setPanPos(float pos);
    virtual void setTiltPos(float pos);
    virtual void setGimbalPos(float panPos,float tiltPos);
    virtual void setEOZoom(QString command, float value);
    virtual void setIRZoom(QString command);
    virtual void snapShot();
    virtual void changeTrackSize(float trackSize);
    virtual void setDigitalStab(bool enable);
    virtual void setRecord(bool enable);
    virtual void setShare(bool enable);
    virtual void setGimbalMode(QString mode);
    virtual void setGimbalPreset(QString mode);
    virtual void setGimbalRecorder(bool enable);
    virtual void setGCSRecorder(bool enable);
    virtual void setLockMode(QString mode,QPoint location = QPoint(0,0));
    virtual void setGeoLockPosition(QPoint location);

    virtual void handleGimbalModeChanged(QString mode);
    virtual void handleGimbalSetModeFail();
protected:
    GimbalData* m_context = nullptr;
    VideoEngine* m_videoEngine = nullptr;
    bool m_isGimbalConnected = false;
    Config* m_config = nullptr;
    JoystickThreaded*  m_joystick = nullptr;
    TargetLocalization* m_targetLocation = nullptr;
protected:

    void  resetTrackParam(){
        m_iPan = 0.0;
        m_cPan = 0.0;
        m_dPanOld = 0.0;
        m_panRate = 0.0;
        m_uPan = 0.0;
        m_iTilt = 0.0;
        m_cTilt = 0.0;
        m_dTiltOld = 0.0;
        m_tiltRate = 0.0;
        m_uTilt = 0.0;
    }
    // tracker param
    double m_iPan= 0 ;
    double m_cPan= 0 ;
    double m_dPanOld= 0 ;
    double m_panRate= 0 ;
    double m_uPan= 0 ;
    double m_kpPan = 1.4;
    double m_kiPan = 0.02;//1.0
    double m_kdPan = 0;
    double m_iTilt= 0 ;
    double m_cTilt= 0 ;
    double m_dTiltOld= 0 ;
    double m_tiltRate= 0 ;
    double m_uTilt= 0 ;
    double m_kpTilt = 1.3;
    double m_kiTilt = 0.02;
    double m_kdTilt= 0.0;

    clock_t m_beginTime=0;
    clock_t m_endTime=0;
};

#endif // GIMBALINTERFACE_H
