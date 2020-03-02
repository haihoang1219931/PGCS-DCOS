#include "samplegimbal.h"
SampleGimbal::SampleGimbal(QObject *parent) : QObject(parent)
{
    m_buffer = new BufferOut();
    _receiveSocket = new QUdpSocket();
    _gimbalModel = new GimbalInterfaceContext();
    _packetParser = new GimbalPacketParser();
    _ipcCommands = new IPCCommands();
    _systemCommands = new SystemCommands();
    _motionCCommands = new MotionCCommands();
    _geoCommands = new GeoCommands();
    m_epSensorTool = new EPHucomTool;
}
SampleGimbal::~SampleGimbal()
{
    m_buffer->deleteLater();
    _receiveSocket->deleteLater();
    _packetParser->deleteLater();
    _gimbalModel->deleteLater();
    _ipcCommands->deleteLater();
    _systemCommands->deleteLater();
    _motionCCommands->deleteLater();
}
void SampleGimbal::newConnect(string gimbalAddress, int receivePort,
                              int listenPort)
{
    SampleGimbal::newConnect(QString::fromStdString(gimbalAddress), receivePort,
                             listenPort);
}

void SampleGimbal::newConnect(QString gimbalAddress, int receivedPort,
                              int listenPort)
{
    _packetParser->Reset();
    // Init tcp sensor connect
    m_tcpSensor = new ClientStuff("192.168.0.103", 11999);
    connect(m_tcpSensor, &ClientStuff::hasReadSome, this, &SampleGimbal::_onTCPSensorReceivedData);
    connect(m_tcpSensor, &ClientStuff::statusChanged, this, &SampleGimbal::_onTCPSensorStatusChanged);
    connect(m_tcpSensor->tcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
                this, SLOT(gotError(QAbstractSocket::SocketError)));
    m_tcpSensor->connect2host();
    _ipcCommands->m_tcpSensor = m_tcpSensor;
    // Init tcp gimbal connect
    m_tcpGimbal = new ClientStuff("192.168.0.123", 1234);
    connect(m_tcpGimbal, &ClientStuff::hasReadSome, this, &SampleGimbal::_onTCPGimbalReceivedData);
    connect(m_tcpGimbal, &ClientStuff::statusChanged, this, &SampleGimbal::_onTCPGimbalStatusChanged);
    connect(m_tcpGimbal->tcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
                this, SLOT(gotError(QAbstractSocket::SocketError)));
    m_tcpGimbal->connect2host();
    _ipcCommands->m_tcpGimbal = m_tcpGimbal;
    //_sendSocket = new QUdpSocket();
    m_buffer->setIP(gimbalAddress.toStdString());
    m_buffer->setPort(receivedPort);
    m_buffer->init();
    //_sendSocket->connectToHost(gimbalAddress,receivedPort,QUdpSocket::ReadWrite);
    _receiveSocket->bind(QHostAddress::Any, listenPort);
    connect(_receiveSocket, SIGNAL(readyRead()), this,
            SLOT(_receiveSocket_PacketReceived()));
    connect(_packetParser, SIGNAL(UavvGimbalPacketParsed(key_type, vector<byte>)),
            this, SLOT(PacketParser_PacketReceived(key_type, vector<byte>)));
    _ipcCommands->m_buffer = m_buffer;
    _systemCommands->m_buffer = m_buffer;
    _motionCCommands->m_buffer = m_buffer;
    _geoCommands->m_buffer = m_buffer;
    _ipcCommands->m_gimbalModel = _gimbalModel;
    _systemCommands->m_gimbalModel = _gimbalModel;
    _motionCCommands->m_gimbalModel = _gimbalModel;
    _geoCommands->m_gimbalModel = _gimbalModel;
    /**/
    connect(_gimbalModel->m_gimbal, SIGNAL(NotifyPropertyChanged(QString)), this,
            SLOT(changeGimbalInfo(QString)));
    PacketSetup();
    GimbalStateSetup();
    m_isGimbalConnected = true;
}

void SampleGimbal::newDisconnect()
{
    m_buffer->uinit();
    _receiveSocket->close();
    m_isGimbalConnected = false;
}

void SampleGimbal::newInitialise() {}

void SampleGimbal::PacketSetup() {}
void SampleGimbal::GimbalStateSetup() {}

void SampleGimbal::ResetCommsWatchDog() {}

void SampleGimbal::ParseGimbalPacket(key_type _key, vector<byte> _data)
{
    //      printf("\nKey: %d\n Data: ", _key);
    //    for(unsigned int i = 0; i < _data.size(); i++){
    //        printf("0x%02X ", _data[i]);
    //    }
    switch (_key) {
    case (key_type)EyePhoenixProtocol::Confirm: {
        Confirm data;
        data.parse(_data);
        key_type key = data.getKey();
        break;
    }

    case (key_type)EyePhoenixProtocol::ErrorMessage: {
    }

    case (key_type)EyePhoenixProtocol::Telemetry: {
        //            Telemetry tele;
        //            tele.parse(_data);
        //            printf("\nRecevied gps data = [%f, %f, %f]", tele.getPn(),
        //            tele.getPe(), tele.getPd());
        //            _gimbalModel->m_gimbal->updateGPSData(tele.getPn(),
        //            tele.getPe(), tele.getPd(),
        //                                                  tele.getRoll(),
        //                                                  tele.getPitch(),
        //                                                  tele.getYaw());
        break;
    }

    case (key_type)EyePhoenixProtocol::IPCStatusResponse: {
        break;
    }

    case (key_type)EyePhoenixProtocol::TargetPosition: {
        TargetPosition targetPosition;
        targetPosition.parse(_data);
        // TODO: Compare received status with current status. If this state is
        // different from current state, through an ERROR
        _gimbalModel->m_gimbal->setCorners(
            targetPosition.getApn(), targetPosition.getApe(),
            targetPosition.getBpn(), targetPosition.getBpe(),
            targetPosition.getCpn(), targetPosition.getCpe(),
            targetPosition.getDpn(), targetPosition.getDpe(),
            targetPosition.getOpn(), targetPosition.getOpe());
        break;
    }

    case (key_type)EyePhoenixProtocol::GeolockInfo: {
        GPSData geolockPoint;
        geolockPoint.parse(_data);
        // TODO: Compare received status with current status. If this state is
        // different from current state, through an ERROR
        _gimbalModel->m_gimbal->setGeolockLocation(
            geolockPoint.getPn(), geolockPoint.getPe(), geolockPoint.getPd());
        break;
    }

    case (key_type)EyePhoenixProtocol::MotionCStatus: {
        Eye::MotionCStatus motionCState;
        motionCState.parse(_data);
        // TODO: Compare received status with current status. If this state is
        // different from current state, through an ERROR
        _gimbalModel->m_gimbal->updateMotionCStatus(
            motionCState.getPanStabMode() == (byte)Status::StabMode::ON &&
            motionCState.getTiltStabMode() == (byte)Status::StabMode::ON,
            motionCState.getPanPos(), motionCState.getTiltPos(),
            motionCState.getPanVelo(), motionCState.getTiltVelo());
        break;
    }

    case (key_type)EyePhoenixProtocol::SystemStatus: {
        Eye::SystemStatus systemState;
        systemState.parse(_data);
        //            printf("systemState.getIndex() =
        //            %d\r\n",systemState.getIndex());
        Eye::MotionCStatus motionCStatus = systemState.getMotionCStatus();
        _gimbalModel->m_gimbal->updateMotionCStatus(motionCStatus);
        Eye::IPCStatusResponse ipcStatusResponse = systemState.getIPCStatus();
        // TODO: Compare received status with current status. If this state is
        // different from current state, through an ERROR
        //                printf("\nTrackSize = %f", ipcStatusResponse.getTrackSize());
        _gimbalModel->m_gimbal->updateIPCState(ipcStatusResponse);
        _gimbalModel->m_gimbal->m_rbSystem->add(systemState);
        Telemetry tele = systemState.getTelemetry();
        //            printf("\nRecevied gps data = [%f, %f, %f]", tele.getPn(),
        //            tele.getPe(), tele.getPd());
        _gimbalModel->m_gimbal->updateGPSData(
            tele.getPn(), tele.getPe(), tele.getPd(), tele.getRoll() / PI * 180.0f,
            tele.getPitch() / PI * 180.0f, tele.getYaw() / PI * 180.0f);
        char geodata[1024];
        // lat,lon.alt,roll,pitch,yaw,az,el
        sprintf(geodata, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", tele.getPn(),
                tele.getPe(), tele.getPd(), tele.getRoll() / PI * 180.0f,
                tele.getPitch() / PI * 180.0f, tele.getYaw() / PI * 180.0f,
                motionCStatus.getPanPos() / PI * 180.0f,
                motionCStatus.getTiltPos() / PI * 180.0f, tele.getSpeedNorth(),
                tele.getSpeedEast(), tele.getGPSAlt(), tele.getTakeOffAlt());
        //            printf("\n===> %s", geodata);
        //            printf("[az,el] =
        //            %f,%f\r\n",motionCStatus.getPanPos()/PI*180.0f,motionCStatus.getTiltPos()/PI*180.0f);
        //            FileControler::addLine("geodata.txt",std::string(geodata));
        break;
    }

    case (key_type)EyePhoenixProtocol::EOSteeringResponse: {
        Eye::XPoint xPoint;
        xPoint.parse(_data);
        //        printf("\nDelta: Steering Point: [%d]-[%f, %f, %f, %f]",
        //               xPoint.getIndex(), xPoint.getPx(), xPoint.getPy(),
        //               xPoint.getWidth(), xPoint.getHeight());
        _gimbalModel->m_gimbal->m_rbXPointEO->add(xPoint);
        break;
    }

    case (key_type)EyePhoenixProtocol::EOTrackingResponse: {
        Eye::TrackResponse trackRes;
        trackRes.parse(_data);
        //        printf("\nDelta: Track Point: [%d]-[%f, %f, %f, %f, %f, %f]",
        //               trackRes.getIndex(), trackRes.getPx(), trackRes.getPy(),
        //               trackRes.getWidth(), trackRes.getHeight(),
        //               trackRes.getObjWidth(), trackRes.getObjHeight());
        _gimbalModel->m_gimbal->m_rbTrackResEO->add(trackRes);
        break;
    }

    case (key_type)EyePhoenixProtocol::IRSteeringResponse: {
        Eye::XPoint xPoint;
        xPoint.parse(_data);
        //            printf("\nDelta: Steering Point: [%d]-[%f, %f, %f, %f]",
        //            xPoint.getIndex(), xPoint.getPx(), xPoint.getPy(),
        //            xPoint.getWidth(), xPoint.getHeight());
        _gimbalModel->m_gimbal->m_rbXPointIR->add(xPoint);
        break;
    }

    case (key_type)EyePhoenixProtocol::IRTrackingResponse: {
        Eye::TrackResponse trackRes;
        trackRes.parse(_data);
        //            printf("\nDelta: Track Point: [%d]-[%f, %f, %f, %f, %f,
        //            %f]",trackRes.getIndex(), trackRes.getPx(), trackRes.getPy(),
        //            trackRes.getWidth(), trackRes.getHeight(),
        //            trackRes.getObjWidth(), trackRes.getObjHeight());
        _gimbalModel->m_gimbal->m_rbTrackResIR->add(trackRes);
        break;
    }

    case (key_type)EyePhoenixProtocol::GLValid: {
        GPSData gpsData;
        gpsData.parse(_data);
        _gimbalModel->m_gimbal->updateGLMeasured(gpsData.getPn(), gpsData.getPe(),
                gpsData.getPd());
        gimbalInfoChanged("GLValid");
        break;
    }

    case (key_type)EyePhoenixProtocol::GLInvalid: {
        GPSData gpsData;
        gpsData.parse(_data);
        _gimbalModel->m_gimbal->updateGLMeasured(gpsData.getPn(), gpsData.getPe(),
                gpsData.getPd());
        gimbalInfoChanged("GLInvalid");
        break;
    }

    case (key_type)EyePhoenixProtocol::EOMotionDataResponse: {
        Eye::MotionImage motionData;
        motionData.parse(_data);
        _gimbalModel->m_gimbal->m_rbIPCEO->add(motionData);
        break;
    }

    case (key_type)EyePhoenixProtocol::IRMotionDataResponse: {
        Eye::MotionImage motionData;
        motionData.parse(_data);
        _gimbalModel->m_gimbal->m_rbIPCIR->add(motionData);
        break;
    }

    default: {
        break;
    }
    }
}

void SampleGimbal::_receiveSocket_PacketReceived()
{
    while (_receiveSocket->hasPendingDatagrams()) {
        QHostAddress ip = QHostAddress::Any; // any ipaddress
        quint16 port = 0;
        unsigned char receivedData[1024];
        // qint64 received =
        // _receiveSocket->read((char*)receivedData,sizeof(receivedData));
        qint64 received = _receiveSocket->readDatagram(
                              (char *)receivedData, sizeof(receivedData), &ip, &port);

        if (received > 0) {
            _packetParser->Push(receivedData, received);
            _packetParser->Parse();
        }
    }

    // qDebug()<<"Reading data done";
}

void SampleGimbal::changeGimbalInfo(QString name)
{
    Q_EMIT gimbalInfoChanged(name);
}
void SampleGimbal::PacketParser_PacketReceived(key_type key,
        vector<byte> data)
{
    ParseGimbalPacket(key, data);
}
void SampleGimbal::forwardTelePacketReceived(int time, float alt, float lat,
        float lon, float psi, float theta,
        float phi, float windHead,
        float windSpeed, float airSpeed,
        float groundSpeed, float track,
        float amsl, float gps)
{
    //_geoCommand->sendExternalPosition(lat,lon,alt);
    //_geoCommand->sendExternalAltitude(phi,theta,psi);
}

void SampleGimbal::_onTCPSensorStatusChanged(bool _newStatus)
{
    //qDebug() << "new status is:" << newStatus;
    if (_newStatus) {
        Q_EMIT tcpStatusChanged("CONNECTED");
    } else {
        Q_EMIT tcpStatusChanged("DISCONNECTED");
    }
}

bool SampleGimbal::isTCPSensorConnected()
{
    return m_tcpSensor->tcpSocket->isOpen();
}

void SampleGimbal::_onTCPSensorReceivedData(QByteArray _msg){
//    printf("\nTCP Sensor Received Data: ");
//    for(int i = 0; i < _msg.size(); i++){
//        printf("0x%02X, ", (unsigned char)_msg.at(i));
//    }
//    printf("\n====\nZoomPos");
    if((_msg.size() == 7)){
        if((_msg.at(0) == (char)0x90) && (_msg.at(1) == (char)0x50) && (_msg.at(6) == (char)0xFF)){
            std::vector<byte> dataSent(_msg.data() + 2, _msg.data() + 6);
//            for(int i = 0; i < dataSent.size(); i++){
//                printf("0x%02X, ", (unsigned char)dataSent.at(i));
//            }
            uint16_t zoomPosInt = m_epSensorTool->dataSend2ZoomPos(dataSent);
            _gimbalModel->m_gimbal->m_hfov[0] = m_epSensorTool->zoomPos2Fov(zoomPosInt);
//            printf("\nHFOV = %f", _gimbalModel->m_gimbal->m_hfov[0]);
        }
    }
}

void SampleGimbal::_onTCPGimbalStatusChanged(bool _newStatus)
{
    //qDebug() << "new status is:" << newStatus;
    if (_newStatus) {
        Q_EMIT tcpStatusChanged("CONNECTED");
    } else {
        Q_EMIT tcpStatusChanged("DISCONNECTED");
    }
}

bool SampleGimbal::isTCPGimbalConnected()
{
    return m_tcpSensor->tcpSocket->isOpen();
}

void SampleGimbal::_onTCPGimbalReceivedData(QByteArray _msg){
//    printf("\nTCP Gimbal Received Data: ");
//    for(int i = 0; i < _msg.size(); i++){
//        printf("0x%02X, ", (unsigned char)_msg.at(i));
//    }
}
