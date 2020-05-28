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
Q_SIGNALS:

public Q_SLOTS:
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
    void setGimbalMode(QString mode) override;
    void setGimbalPreset(QString mode) override;
    void setGimbalRecorder(bool enable) override;
    void setLockMode(QString mode, QPoint location=QPoint(0,0)) override;
    void setGeoLockPosition(QPoint location) override;
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
    GimbalPacketParser *_packetParser;
    UavvGimbalProtocolSystemPackets *_systemCommand;
    UavvGimbalProtocolGimbalPackets *_gimbalCommand;
    UavvGimbalProtocolEOSensorPackets *_eoCommand;
    UavvGimbalProtocolVideoProcessorPackets *_videoCommand;
    UavvGimbalProtocolIRSensorPackets *_irCommand;
    UavvGimbalProtocolLaserRangeFinderPackets *_lrfCommand;
    UavvGimbalProtocolGeoPointingPackets *_geoCommand;
};

#endif // CM160GIMBAL_H
