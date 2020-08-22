#ifndef TRERONGIMBAL_H
#define TRERONGIMBAL_H

#include <QObject>
#include "../GimbalInterface.h"
#include "Packet/Confirm.h"
#include "Payload/Buffer/BufferOut.h"
#include "Command/GeoCommands.h"
#include "Command/SystemCommands.h"
#include "Command/IPCCommands.h"
#include "Command/MotionCCommands.h"
class TreronGimbalPacketParser;
class TreronGimbal : public GimbalInterface
{
    Q_OBJECT
public:
    explicit TreronGimbal(GimbalInterface *parent = nullptr);

    void connectToGimbal(Config* config = nullptr) override;
    void disconnectGimbal() override;
    void discoverOnLan() override;
    void setPanRate(float rate) override;
    void setTiltRate(float rate) override;
    void setGimbalRate(float panRate,float tiltRate) override;
    void setPanPos(float pos) override;
    void setTiltPos(float pos) override;
    void setGimbalPos(float panPos,float tiltPos) override;
    void setEOZoom(QString command, float value) override;
    void setIRZoom(QString command) override;
    void changeSensor(QString sensorID) override;
    void snapShot() override;
    void changeTrackSize(float trackSize) override;
    void setDigitalStab(bool enable) override;
    void setGimbalMode(QString mode) override;
    void setGimbalPreset(QString mode) override;
    void setGimbalRecorder(bool enable) override;
    void setLockMode(QString mode, QPointF location=QPointF(0,0)) override;
    void setGeoLockPosition(QPoint location) override;
    void setRecord(bool enable) override;
    void setShare(bool enable) override;
Q_SIGNALS:

public Q_SLOTS:
    void handlePacketReceived();
    void handlePacketParsed(key_type key, vector<byte> data);
private:
    BufferOut* m_buffer;
    QUdpSocket *_receiveSocket;           // to recevie packets from gimbal
    TreronGimbalPacketParser *_packetParser;
    IPCCommands *_ipcCommands;
    SystemCommands *_systemCommands;
    MotionCCommands *_motionCCommands;
    GeoCommands *_geoCommands;
};

#endif // TRERONGIMBAL_H
