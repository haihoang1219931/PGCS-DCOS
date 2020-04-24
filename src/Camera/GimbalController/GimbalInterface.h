#ifndef GIMBALINTERFACE_H
#define GIMBALINTERFACE_H

#include <QObject>
#include <QString>
#include <QPoint>
#include "GimbalData.h"
#include "Setting/config.h"
class GimbalInterface : public QObject
{
    Q_OBJECT
public:
    explicit GimbalInterface(QObject *parent = nullptr);
    GimbalData* context(){ return m_context; }
Q_SIGNALS:

public Q_SLOTS:
    virtual void connectToGimbal(Config* config = nullptr);
    virtual void disconnectGimbal();
    virtual void discoverOnLan();
    virtual void setPanRate(float rate);
    virtual void setTiltRate(float rate);
    virtual void setGimbalRate(float panRate,float tiltRate);
    virtual void setPanPos(float pos);
    virtual void setTiltPos(float pos);
    virtual void setGimbalPos(float panPos,float tiltPos);
    virtual void setEOZoom(QString command, QString value);
    virtual void setIRZoom(QString command);
    virtual void snapShot();
    virtual void setGimbalMode(QString mode);
    virtual void setGimbalPreset(QString mode);
    virtual void setGimbalRecorder(QString mode);
    virtual void setLockMode(QString mode, bool enable, QPoint location);
    virtual void setGeoLockPosition(QPoint location);
public:
    GimbalData* m_context;
    bool m_isGimbalConnected = false;
    Config* m_config = nullptr;
};

#endif // GIMBALINTERFACE_H
