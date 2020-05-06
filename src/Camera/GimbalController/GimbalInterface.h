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
public:
    explicit GimbalInterface(QObject *parent = nullptr);
    GimbalData* context(){ return m_context; }
    void setVideoEngine(VideoEngine* videoEngine);
    JoystickThreaded* joystick();
    virtual void setJoystick(JoystickThreaded* joystick);
Q_SIGNALS:
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
    virtual void setEOZoom(QString command, int value);
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
    float m_iPan= 0 ;
    float m_cPan= 0 ;
    float m_dPanOld= 0 ;
    float m_panRate= 0 ;
    float m_uPan= 0 ;

    float m_kpPan = 45.0f;
    float m_kiPan = 1.0;
    float m_kdPan = 0.05;

    float m_iTilt= 0 ;
    float m_cTilt= 0 ;
    float m_dTiltOld= 0 ;
    float m_tiltRate= 0 ;
    float m_uTilt= 0 ;

    float m_kpTilt = 50.0f;
    float m_kiTilt = 5.0;
    float m_kdTilt= 0.05f;
};

#endif // GIMBALINTERFACE_H
