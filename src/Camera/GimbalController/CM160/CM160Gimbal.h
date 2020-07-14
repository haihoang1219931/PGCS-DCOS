#ifndef CM160GIMBAL_H
#define CM160GIMBAL_H

#include <QObject>
#include "../GimbalInterface.h"
#include "GimbalDiscoverer.h"
#include "GimbalPacketParser.h"
#include <QUdpSocket>
#include <QTimer>
#include "UavvGimbalProtocol.h"

#include "uavvgimbalprotocolsystempackets.h"
#include "uavvgimbalprotocolgimbalpackets.h"
#include "uavvgimbalprotocoleosensorpackets.h"
#include "uavvgimbalprotocolvideoprocessorpackets.h"
#include "uavvgimbalprotocolirsensorpackets.h"
#include "uavvgimbalprotocollaserrangefinderpackets.h"
#include "uavvgimbalprotocolgeopointingpackets.h"
class CM160Gimbal : public GimbalInterface
{
    Q_OBJECT
public:
    explicit CM160Gimbal(GimbalInterface *parent = nullptr);
    ~CM160Gimbal();

public:
    Q_INVOKABLE void connectToGimbal(Config* config = nullptr) override;
    Q_INVOKABLE void disconnectGimbal() override;
    Q_INVOKABLE void discoverOnLan() override;
    Q_INVOKABLE void setPanRate(float rate) override;
    Q_INVOKABLE void setTiltRate(float rate) override;
    Q_INVOKABLE void setGimbalRate(float panRate,float tiltRate) override;
    Q_INVOKABLE void setPanPos(float pos) override;
    Q_INVOKABLE void setTiltPos(float pos) override;
    Q_INVOKABLE void setGimbalPos(float panPos,float tiltPos) override;
    Q_INVOKABLE void setEOZoom(QString command, float value) override;
    Q_INVOKABLE void setIRZoom(QString command) override;
    Q_INVOKABLE void changeSensor(QString sensorID) override;
    Q_INVOKABLE void snapShot() override;
    Q_INVOKABLE void setGimbalMode(QString mode) override;
    Q_INVOKABLE void setGimbalPreset(QString mode) override;
    Q_INVOKABLE void setGimbalRecorder(bool enable) override;
    Q_INVOKABLE void setLockMode(QString mode, QPointF location= QPointF(0,0)) override;
    Q_INVOKABLE void setGeoLockPosition(QPoint location) override;
Q_SIGNALS:
public Q_SLOTS:   
    void handlePacketReceived();
    void handlePacketParsed(GimbalPacket packet,unsigned char checksum);
    void requestData();
private:
    void PacketSetup();
    void ParseGimbalPacket(GimbalPacket packet);
    void ParseVersionPacket(GimbalPacket packet);
    void ParseCombinedPositionVelocityState(GimbalPacket packet);
    void ParseZoomPosition(GimbalPacket packet);
    void ParseSensorFOV(GimbalPacket packet);
    void ParseCurrentGimbalMode(GimbalPacket packet);
    void ParseCornerLocations(GimbalPacket packet);
    void ParseCurrentGeolockLocations(GimbalPacket packet);
    void ParseCurrentTargetLocations(GimbalPacket packet);
    void ParsePositionPacket(GimbalPacket packet);
    void ParsePlatformPacket(GimbalPacket packet);
    void ParsePlatformOffsetPacket(GimbalPacket packet);
    void ParseLaserRange(GimbalPacket packet);
    void ParseTrackingParams(GimbalPacket packet);
    void ParseRecordingStatus(GimbalPacket packet);
    void ParsePrimarySensor(GimbalPacket packet);
    void ParseGyroStab(GimbalPacket packet);
    void ParseEStab(GimbalPacket packet);
    void ParseStowMode(GimbalPacket packet);
    void ParseTrackMode(GimbalPacket packet);
    void ParseOverlay(GimbalPacket packet);
    void ParseVideoDestination(GimbalPacket packet);
private:
    QTimer m_timerRequest;
    GimbalDiscoverer* m_gimbalDiscover;
    QUdpSocket *_sendSocket; // to send gimbal packets to gimbal
    QUdpSocket *_receiveSocket; // to recevie packets from gimbal
    CM160GimbalPacketParser *_packetParser;
    UavvGimbalProtocolSystemPackets *_systemCommand;
    UavvGimbalProtocolGimbalPackets *_gimbalCommand;
    UavvGimbalProtocolEOSensorPackets *_eoCommand;
    UavvGimbalProtocolVideoProcessorPackets *_videoCommand;
    UavvGimbalProtocolIRSensorPackets *_irCommand;
    UavvGimbalProtocolLaserRangeFinderPackets *_lrfCommand;
    UavvGimbalProtocolGeoPointingPackets *_geoCommand;
};

#endif // CM160GIMBAL_H
