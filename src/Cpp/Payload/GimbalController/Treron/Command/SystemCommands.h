#ifndef SYSTEMCOMMANDS_H
#define SYSTEMCOMMANDS_H

#include <QQuickItem>
#include <QSocketNotifier>
#include <QUdpSocket>

#include "Payload/Buffer/BufferOut.h"
#include "../../GimbalData.h"

#include "../Packet/EyephoenixProtocol.h"
#include "../Packet/KLV.h"
#include "../Packet/InstallMode.h"
#include "../Packet/RequestResponsePacket.h"
#include "../Packet/RapidView.h"

class SystemCommands: public QObject
{
    Q_OBJECT
public:
    SystemCommands(QObject* parent = 0);
    virtual ~SystemCommands();
public:
    Q_INVOKABLE void setInstallMode(QString mouseMode, QString apMode);
    Q_INVOKABLE void getCameraStatus();
    Q_INVOKABLE void resetIMU();
    Q_INVOKABLE void calibIMU();
public:
    BufferOut* m_buffer;
    GimbalData* m_gimbalModel;
};

#endif
