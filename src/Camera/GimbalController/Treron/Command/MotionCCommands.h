#ifndef MOTIONCCOMMANDS_H
#define MOTIONCCOMMANDS_H

#include <QQuickItem>
#include <QSocketNotifier>
#include <QUdpSocket>

#include "../Packet/EyephoenixProtocol.h"
#include "../Packet/KLV.h"
#include "../Packet/RapidView.h"
#include "../Packet/PTRateFactor.h"
#include "../Packet/PTAngleDiff.h"
#include "../Packet/PTAngle.h"
#include "../Packet/GimbalStab.h"
#include "../Packet/EyeStatus.h"
#include "../Packet/MCParams.h"
#include "Camera/Buffer/BufferOut.h"
#include "../../GimbalData.h"
class MotionCCommands: public QObject
{
    Q_OBJECT
public:
    MotionCCommands(QObject* parent = 0);
    virtual ~MotionCCommands();
public:
    Q_INVOKABLE void changeRapidView(QString mode);
    Q_INVOKABLE void setPanTiltVelocity(int id, float panVel, float tiltVel);
    Q_INVOKABLE void setPanTiltAngle(float panAngle, float tiltAngle);
    Q_INVOKABLE void setPanTiltDiffAngle(float panDif, float tiltDif);
    Q_INVOKABLE void enableGimbalStab(bool stabPan, bool stabTilt);
    Q_INVOKABLE void setMCPanParams(float _kp,float _ki);
    Q_INVOKABLE void setMCTiltParams(float _kp,float _ki);
public:
    BufferOut *m_buffer;
    GimbalData* m_gimbalModel;
};

#endif
