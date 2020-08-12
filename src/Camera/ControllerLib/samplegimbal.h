#ifndef NETWORK_H
#define NETWORK_H

#include "gimbalinterfacecontext.h"
#include "gimbalpacketparser.h"
#include <QQuickItem>
#include <QSocketNotifier>
#include <QTimer>
#include <QUdpSocket>

#include "Buffer/BufferOut.h"
#include "Packet/Common_type.h"
#include "Packet/Confirm.h"
#include "Packet/EOS.h"
#include "Packet/EyeCheck.h"
#include "Packet/EyeEvent.h"
#include "Packet/EyeStatus.h"
#include "Packet/GPSData.h"
#include "Packet/GPSRate.h"
#include "Packet/GimbalMode.h"
#include "Packet/GimbalRecord.h"
#include "Packet/GimbalRecordStatus.h"
#include "Packet/GimbalStab.h"
#include "Packet/IPCStatusResponse.h"
#include "Packet/ImageStab.h"
#include "Packet/InstallMode.h"
#include "Packet/KLV.h"
#include "Packet/LockMode.h"
#include "Packet/Matrix.h"
#include "Packet/MotionAngle.h"
#include "Packet/MotionCStatus.h"
#include "Packet/Object.h"
#include "Packet/PTAngleDiff.h"
#include "Packet/PTRateFactor.h"
#include "Packet/RFData.h"
#include "Packet/RFRequest.h"
#include "Packet/RTData.h"
#include "Packet/RapidView.h"
#include "Packet/ScreenPoint.h"
#include "Packet/SensorColor.h"
#include "Packet/SensorId.h"
#include "Packet/Snapshot.h"
#include "Packet/TargetPosition.h"
#include "Packet/Telemetry.h"
#include "Packet/Vector.h"
#include "Packet/XPoint.h"
#include "Packet/ZoomData.h"
#include "Packet/ZoomStatus.h"
#include "Packet/utils.h"

#include "Command/GeoCommands.h"
#include "Command/IPCCommands.h"
#include "Command/MotionCCommands.h"
#include "Command/SystemCommands.h"
#include "Packet/MotionImage.h"
#include "Packet/SystemStatus.h"
#include "Packet/TrackResponse.h"

#include "tcp/clientStuff.h"

#include "EPTools/EPHucomTool.h"


using namespace Eye;
class SampleGimbal : public QObject
{
        Q_OBJECT
        Q_PROPERTY(GimbalInterfaceContext *gimbalModel READ gimbalModel)
        Q_PROPERTY(bool isGimbalConnected READ isGimbalConnected)
        Q_PROPERTY(bool isSensorConnected READ isSensorConnected)

        Q_PROPERTY(IPCCommands *ipcCommands READ ipcCommands)
        Q_PROPERTY(SystemCommands *systemCommands READ systemCommands)
        Q_PROPERTY(MotionCCommands *montionCCommands READ motionCCommands)
        Q_PROPERTY(GeoCommands *geoCommands READ geoCommands)

        Q_PROPERTY(QString logFolder READ logFolder)

    public:
        SampleGimbal(QObject *parent = 0);
        virtual ~SampleGimbal();
        QTimer _detectGimbalTimer; // to detect that comms have been established with
        // gimbal
        QTimer _communicationWatchdog; // to detect if comms have been lost with the
        // gimbal
        GimbalInterfaceContext *_gimbalModel; // keep the state of the gimbal
        BufferOut *m_buffer;                  // to send gimbal packets to gimbal
        QUdpSocket *_receiveSocket;           // to recevie packets from gimbal
        GimbalPacketParser *_packetParser;
        ClientStuff *m_tcpSensor = nullptr;
        ClientStuff *m_tcpGimbal = nullptr;

        std::string m_logFolder = "";
        const int DETECT_GIMBAL_PERIOD = 1000;  // milliseconds
        const int COMMS_WATCHDOG_PERIOD = 5000; // milliseconds
        bool m_isGimbalConnected = false;
        EPSensorTool *m_epSensorTool;


        bool isSensorConnected()
        {
            return (m_tcpSensor != nullptr) && (m_tcpSensor->tcpSocket->isOpen());
        }

        bool isGimbalConnected()
        {
            return (m_tcpGimbal != nullptr) && (m_tcpGimbal->tcpSocket->isOpen());
        }

        QString logFolder()
        {
            return QString::fromStdString(m_logFolder);
        }

        IPCCommands *_ipcCommands;
        SystemCommands *_systemCommands;
        MotionCCommands *_motionCCommands;
        GeoCommands *_geoCommands;

        GimbalInterfaceContext *gimbalModel()
        {
            return _gimbalModel;
        }

        IPCCommands *ipcCommands()
        {
            return _ipcCommands;
        }

        SystemCommands *systemCommands()
        {
            return _systemCommands;
        }

        MotionCCommands *motionCCommands()
        {
            return _motionCCommands;
        }

        GeoCommands *geoCommands()
        {
            return _geoCommands;
        }

        Q_INVOKABLE void newConnect(string gimbalAddress, int receivePort,
                                    int listenPort);
        Q_INVOKABLE void newConnect(QString gimbalAddress, int receivePort,
                                    int listenPort);
        Q_INVOKABLE void newDisconnect();
        Q_INVOKABLE void newInitialise();
        Q_INVOKABLE void forwardTelePacketReceived(int time, float alt, float lat,
                float lon, float psi, float theta,
                float phi, float windHead,
                float windSpeed, float airSpeed,
                float groundSpeed, float track,
                float amsl, float gps);

        void PacketSetup();
        void GimbalStateSetup();
        void ResetCommsWatchDog();
        void ParseGimbalPacket(key_type key, vector<byte> data);
    Q_SIGNALS:
        // void gimbalModelChanged();
        void gimbalInfoChanged(QString name);
        void tcpStatusChanged(QString newStatus);

    public Q_SLOTS:
        void changeGimbalInfo(QString name);
        void _receiveSocket_PacketReceived();
        void PacketParser_PacketReceived(key_type key, vector<byte> data);
        void _onTCPSensorStatusChanged(bool _newStatus);
        void _onTCPSensorReceivedData(QByteArray _msg);
        bool isTCPSensorConnected();
        void _onTCPGimbalStatusChanged(bool _newStatus);
        void _onTCPGimbalReceivedData(QByteArray _msg);
        bool isTCPGimbalConnected();
};
#endif // NETWORK_H
