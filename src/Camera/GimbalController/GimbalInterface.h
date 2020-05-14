#ifndef GIMBALINTERFACE_H
#define GIMBALINTERFACE_H

#include <QObject>
#include <QString>
#include <QPoint>
#include "GimbalData.h"
#include "Setting/config.h"

class VideoEngine;
class JoystickThreaded;
class GimbalInterface : public QObject
{
    Q_OBJECT
    Q_PROPERTY(JoystickThreaded*    joystick                    READ joystick       WRITE setJoystick)
    Q_PROPERTY(float digitalZoomMax READ digitalZoomMax WRITE setDigitalZoomMax NOTIFY digitalZoomMaxChanged)
    Q_PROPERTY(float zoomMax READ zoomMax WRITE setZoomMax NOTIFY zoomMaxChanged)
    Q_PROPERTY(float zoomMin READ zoomMin WRITE setZoomMin NOTIFY zoomMinChanged)
    Q_PROPERTY(float zoom READ zoom WRITE setZoom NOTIFY zoomChanged)
    Q_PROPERTY(float zoomTarget READ zoomTarget WRITE setZoomTarget NOTIFY zoomTargetChanged)
public:
    explicit GimbalInterface(QObject *parent = nullptr);
    GimbalData* context(){ return m_context; }
    void setVideoEngine(VideoEngine* videoEngine);
    JoystickThreaded* joystick();
    virtual void setJoystick(JoystickThreaded* joystick);
    float digitalZoomMax(){
        return m_context->m_zoomMax[0];
    }
    void setDigitalZoomMax(float digitalZoomMax){
        m_context->m_digitalZoomMax[0] = digitalZoomMax;
        Q_EMIT digitalZoomMaxChanged();
    }
    float zoomMax(){
        return m_context->m_zoomMax[0];
    }
    void setZoomMax(float zoomMax){
        m_context->m_zoomMax[0] = zoomMax;
        Q_EMIT zoomMaxChanged();
    }
    float zoomMin(){
        return m_context->m_zoomMin[0];
    }
    void setZoomMin(float zoomMin){
        m_context->m_zoomMin[0] = zoomMin;
        Q_EMIT zoomMinChanged();
    }
    float zoom(){
        return m_context->m_zoom[0];
    }
    void setZoom(float zoom){
        m_context->m_zoom[0] = zoom;
        Q_EMIT zoomChanged();
    }
    float zoomTarget(){
        return m_context->m_zoomTarget[0];
    }
    void setZoomTarget(float zoomTarget){
        m_context->m_zoomTarget[0] = zoomTarget;
        Q_EMIT zoomTargetChanged();
    }
    float zoomCalculated(){
        return m_context->m_zoomCalculated[0];
    }
    void setZoomCalculated(int index, float zoomCalculated){
        m_context->m_zoomCalculated[index] = zoomCalculated;
        Q_EMIT zoomCalculatedChanged(index,zoomCalculated);
    }
Q_SIGNALS:
    void digitalZoomMaxChanged();
    void zoomMaxChanged();
    void zoomMinChanged();
    void zoomChanged();
    void zoomTargetChanged();
    void zoomCalculatedChanged(int viewIndex,float zoomCalculated);
    void functionHandled(QString message);
public Q_SLOTS:
    virtual void connectToGimbal(Config* config = nullptr);
    virtual void disconnectGimbal();
    virtual void discoverOnLan();
    virtual void changeSensor(QString sensorID);
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
protected:
    GimbalData* m_context = nullptr;
    VideoEngine* m_videoEngine = nullptr;
    bool m_isGimbalConnected = false;
    Config* m_config = nullptr;
    JoystickThreaded*  m_joystick = nullptr;

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
    double m_kpPan = 30.0;
    double m_kiPan = 2;//1.0
    double m_kdPan = 0.3;
    double m_iTilt= 0 ;
    double m_cTilt= 0 ;
    double m_dTiltOld= 0 ;
    double m_tiltRate= 0 ;
    double m_uTilt= 0 ;
    double m_kpTilt = 40.0;
    double m_kiTilt = 2;
    double m_kdTilt= 0.2;

    clock_t m_beginTime=0;
    clock_t m_endTime=0;
};

#endif // GIMBALINTERFACE_H
