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

public Q_SLOTS:
    virtual void connectToGimbal(Config* config = nullptr);
    virtual void disconnectGimbal();
    virtual void discoverOnLan();
    virtual void changeSensor(QString sensorID);
    virtual void handleAxes();
    virtual void setPanRate(float rate);
    virtual void setTiltRate(float rate);
    virtual void setGimbalRate(float panRate,float tiltRate);
    virtual void setPanPos(float pos);
    virtual void setTiltPos(float pos);
    virtual void setGimbalPos(float panPos,float tiltPos);
    virtual void setEOZoom(QString command, int value);
    virtual void setIRZoom(QString command);
    virtual void snapShot();
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
};

#endif // GIMBALINTERFACE_H
