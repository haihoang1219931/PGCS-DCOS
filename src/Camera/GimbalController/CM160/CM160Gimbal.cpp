#include "CM160Gimbal.h"

CM160Gimbal::CM160Gimbal(GimbalInterface *parent) : GimbalInterface(parent)
{
    m_gimbalDiscover = new GimbalDiscoverer();
    _sendSocket = new QUdpSocket();
    _receiveSocket = new QUdpSocket();
    _packetParser = new GimbalPacketParser();
    _systemCommand = new UavvGimbalProtocolSystemPackets();
    _gimbalCommand = new UavvGimbalProtocolGimbalPackets();
    _eoCommand = new UavvGimbalProtocolEOSensorPackets();
    _videoCommand = new UavvGimbalProtocolVideoProcessorPackets();
    _irCommand = new UavvGimbalProtocolIRSensorPackets();
    _lrfCommand = new UavvGimbalProtocolLaserRangeFinderPackets();
    _geoCommand = new UavvGimbalProtocolGeoPointingPackets();
    m_timerRequest.setInterval(100);
    m_timerRequest.setSingleShot(false);
}
CM160Gimbal::~CM160Gimbal()
{
    printf("CM160Gimbal destroyed\r\n");
    if(_sendSocket != nullptr){
        _sendSocket->close();
        _sendSocket->deleteLater();
    }
    if(_receiveSocket != nullptr){
        _receiveSocket->close();
        _receiveSocket->deleteLater();
    }
    _packetParser->deleteLater();
    _systemCommand->deleteLater();
    _gimbalCommand->deleteLater();
    _eoCommand->deleteLater();
    _videoCommand->deleteLater();
    _irCommand->deleteLater();
    _lrfCommand->deleteLater();
    _geoCommand->deleteLater();
}
void CM160Gimbal::connectToGimbal(Config* config){
    if(m_context!=nullptr){
        m_context->m_hfovMin[0] = 2.33f;
        m_context->m_hfovMax[0] = 63.1f;
        m_context->m_zoom[0] = 1;
    }
    if(config != nullptr){
        _sendSocket->connectToHost(config->mapData["GIMBAL_CONTROL_IP"].toString(),
                static_cast<quint16>(config->mapData["GIMBAL_CONTROL_IN"].toInt()));
        _receiveSocket->bind(QHostAddress::AnyIPv4,
                             static_cast<quint16>(config->mapData["GIMBAL_CONTROL_REPLY"].toInt()));
    }else{
        _sendSocket->connectToHost("192.168.0.113",18001);
        _receiveSocket->bind(QHostAddress::AnyIPv4,18002);
    }

    connect(_receiveSocket,SIGNAL(readyRead()),this,SLOT(handlePacketReceived()));
    connect(_packetParser,
            SIGNAL(gimbalPacketParsed(GimbalPacket,unsigned char)),
            this,
            SLOT(handlePacketParsed(GimbalPacket,unsigned char)));
    connect(&m_timerRequest,&QTimer::timeout,this,&CM160Gimbal::requestData);
    _systemCommand->_udpSocket = _sendSocket;
    _gimbalCommand->_udpSocket = _sendSocket;
    _eoCommand->_udpSocket = _sendSocket;
    _videoCommand->_udpSocket = _sendSocket;
    _irCommand->_udpSocket = _sendSocket;
    _lrfCommand->_udpSocket = _sendSocket;
    _geoCommand->_udpSocket = _sendSocket;
    PacketSetup();
    m_isGimbalConnected = true;
    m_timerRequest.start();
}
void CM160Gimbal::disconnectGimbal(){    
    disconnect(&m_timerRequest,&QTimer::timeout,this,&CM160Gimbal::requestData);
    m_timerRequest.stop();
    _sendSocket->close();
    _receiveSocket->close();
    m_isGimbalConnected = false;
}
void CM160Gimbal::PacketSetup(){
    _systemCommand->setSystemTime(time(NULL));
    std::vector<PacketRate> packetRates;
    //stream Pan and Tilt angle at 10Hz
    packetRates.push_back(PacketRate(UavvGimbalProtocol::CombinedPositionVelocityState, 10));
//    packetRates.push_back(PacketRate(UavvGimbalProtocol::SensorFieldOfView, 5));
    packetRates.push_back(PacketRate(UavvGimbalProtocol::TrackingStatus, 5));
    packetRates.push_back(PacketRate(UavvGimbalProtocol::PlatformPosition, 5));
    packetRates.push_back(PacketRate(UavvGimbalProtocol::PlatformOrientation, 10));
    packetRates.push_back(PacketRate(UavvGimbalProtocol::CurrentCornerLocations, 10));
    _systemCommand->requestResponse(int(UavvGimbalProtocol::Version));
    _systemCommand->requestResponse(int(UavvGimbalProtocol::DetailedVersion));
    _systemCommand->requestResponse(int(UavvGimbalProtocol::SetVideoDestination));
    _systemCommand->requestResponse(int(0x73));
    _systemCommand->requestResponse(int(0x00));
    _systemCommand->requestResponse(int(0x11));
    _systemCommand->requestResponse(int(0xc6));
    _systemCommand->requestResponse(int(0xc7));
    _systemCommand->requestResponse(int(0xc9));
    _systemCommand->requestResponse(int(0xd2));
    _systemCommand->requestResponse(int(0x57));
    _systemCommand->requestResponse(int(0x8B));
    unsigned char packet12[] ={0x24,0x40,0x4c,0x0a,0x00,0x7c,0x00,0x00,0x00,0x14,0x00,0x00,0x00,0x03,0xe8,0xc5,0x00,0x00,0x00,0x64,0xc4,0x00,0x00,0x00,0x64,0xc3,0x00,0x00,0x00,0x64,0xba,0x00,0x00,0x00,0x64,0xbc,0x00,0x00,0x00,0x64,0xbb,0x00,0x00,0x00,0x64,0x50,0x00,0x00,0x00,0x64,0xc1,0x00,0x00,0x07,0xd0,0x7b,0x00,0x00,0x03,0xe8,0x13,0x00,0x00,0x00,0x64,0x52,0x00,0x00,0x03,0xe8,0x57,0x00,0x00,0x13,0x88,0x54,0x00,0x00,0x03,0xe8,0xfe};
    _sendSocket->write((const char *)packet12,sizeof(packet12));
    //    _systemCommand->configurePacketRates(packetRates);

//    _gimbalCommand->setSceneSteering(false);
//    _videoCommand->setEStabilisationParameters(false,0,0,0,0);
    _videoCommand->setOverlay(true,false,false,false,false,true,false,false,false,false,false);
    //_eoCommand->enableAutoFocus(true);
    //_eoCommand->enableAutoExposure(true);
    _gimbalCommand->setPrimaryVideo(1,0);
    _eoCommand->setEOOpticalZoomPosition(1);
}
void CM160Gimbal::discoverOnLan(){
    m_gimbalDiscover->requestDiscover();
}
void CM160Gimbal::setPanRate(float rate){
    if(_gimbalCommand!=nullptr){
        _gimbalCommand->setPanVelocity(rate);
    }
}
void CM160Gimbal::setTiltRate(float rate){
    if(_gimbalCommand!=nullptr){
        _gimbalCommand->setTiltVelocity(rate);
    }
}
void CM160Gimbal::setGimbalRate(float panRate,float tiltRate){
    if(_gimbalCommand!=nullptr){
        _gimbalCommand->setPanTiltVelocity(panRate,tiltRate);
    }
}
void CM160Gimbal::setPanPos(float pos){
    if(_gimbalCommand!=nullptr){
        _gimbalCommand->setPanPosition(pos);
    }
}
void CM160Gimbal::setTiltPos(float pos){
    if(_gimbalCommand!=nullptr){
        _gimbalCommand->setTiltPosition(pos);
    }
}
void CM160Gimbal::setGimbalPos(float panPos,float tiltPos){
    if(_gimbalCommand!=nullptr){
        _gimbalCommand->setPanTiltPosition(panPos,tiltPos);
    }
}
void CM160Gimbal::setEOZoom(QString command, int value){
//    Q_UNUSED(command);

    if(_eoCommand!=nullptr){
        if(command != ""){
            _eoCommand->setEOOpticalZoomVelocity(command,value);
        }else
            _eoCommand->setEOOpticalZoomPosition(value);
    }
}
void CM160Gimbal::setIRZoom(QString command){
    Q_UNUSED(command);
}
void CM160Gimbal::changeSensor(QString sensorID){

}
void CM160Gimbal::snapShot(){

}
void CM160Gimbal::setGimbalMode(QString mode){
    Q_UNUSED(mode);
}
void CM160Gimbal::setGimbalPreset(QString mode){
    Q_UNUSED(mode);
}
void CM160Gimbal::setGimbalRecorder(bool enable){
    Q_UNUSED(enable);
}
void CM160Gimbal::setLockMode(QString mode, QPoint location){
    Q_UNUSED(mode);
    Q_UNUSED(location);
}
void CM160Gimbal::setGeoLockPosition(QPoint location){
    Q_UNUSED(location);
}
void CM160Gimbal::handlePacketReceived(){
    while (_receiveSocket->hasPendingDatagrams())
    {
        QHostAddress ip = QHostAddress::Any; //any ipaddress
        quint16 port = 0;
        unsigned char receivedData[1024];
        qint64 received = _receiveSocket->readDatagram(
                    (char*)receivedData,
                    sizeof(receivedData),
                    &ip,
                    &port);
        //printf("Received %d byte from camera\r\n",received);
        if(received>0){
            _packetParser->Push(receivedData,received);
            _packetParser->Parse();
        }

    }
}
void CM160Gimbal::handlePacketParsed(GimbalPacket packet,unsigned char checksum){
//    printf("%s [%d]\r\n",__func__,packet.IDByte);
    switch (packet.IDByte)
    {
        case (unsigned char)UavvGimbalProtocol::Version:
        {
            ParseVersionPacket(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::CombinedPositionVelocityState:
        {
            ParseCombinedPositionVelocityState(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::ZoomPositionResponse:
        {
            ParseZoomPosition(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::SensorFieldOfView:
        {
            ParseSensorFOV(packet);
            break;
        }
        case (unsigned char) UavvGimbalProtocol::TrackingStatus:
        {
            ParseCurrentGimbalMode(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::CurrentCornerLocations:
        {
            ParseCornerLocations(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::CurrentTargetLocation:
        {
            ParseCurrentTargetLocations(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::CurrentGeolockSetpoint:
        {
            ParseCurrentGeolockLocations(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::PlatformPosition:
        {
            ParsePositionPacket(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::PlatformOrientation:
        {
            ParsePlatformPacket(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::PlatformOrientationOffset:
        {
            ParsePlatformOffsetPacket(packet);
            break;
        }
        case (unsigned char)UavvGimbalProtocol::LaserRange:
        {
//            printf("ParseLaserRange\r\n");
            ParseLaserRange(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::SetTrackingParameters:
        {
//            printf("ParseTrackingParams\r\n");
            ParseTrackingParams(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::RecordingStatus:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseRecordingStatus(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::ToggleVideoOutput:
        {
    //            printf("ParseTrackingParams\r\n");
            ParsePrimarySensor(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::SetStowMode:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseStowMode(packet);

        }
        case (unsigned char)UavvGimbalProtocol::ModifyObjectTrack:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseTrackMode(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::GyroStablisation:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseGyroStab(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::SetStabilisationParameters:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseEStab(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::SetOverlayMode:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseOverlay(packet);

        }
        break;
        case (unsigned char)UavvGimbalProtocol::SetVideoDestination:
        {
    //            printf("ParseTrackingParams\r\n");
            ParseVideoDestination(packet);

        }
        break;

        default:
        {

        }
        break;
//            printf("packet ID: %d\r\n",packet.IDByte);
    }
}
void CM160Gimbal::requestData(){
    _eoCommand->getSensorsCurrentFOV();
    _videoCommand->getCurrentRecordingState();
    // check gyro stab mode
    _systemCommand->requestResponse(0);
    // check digital video stab mode
    _systemCommand->requestResponse(114);
    // check current sensor
    _systemCommand->requestResponse(171);
    // check stow mode
    _systemCommand->requestResponse(12);
    // check track size
    _systemCommand->requestResponse(112);
    // check track mode
    _systemCommand->requestResponse(108);
    // check eo color mode
    _systemCommand->requestResponse(41);
    // check ir color mode+
    _systemCommand->requestResponse(167);
    // check eo defog mode
    _systemCommand->requestResponse(44);
    // check geo info
    _systemCommand->requestResponse(186);
    _systemCommand->requestResponse(187);
    _systemCommand->requestResponse(195);
    _systemCommand->requestResponse(196);
    _systemCommand->requestResponse(118);
}
void CM160Gimbal::ParseVersionPacket(GimbalPacket packet){
    //qDebug("Parsing Version Packet");
    UavvVersion versionPacket;
    if (UavvVersion::TryParse(packet,&versionPacket) == ParseResult::Success)
    {
        //qDebug("Valid Version Packet");
        /**/
        QString gimbalSerialNumber = QString::fromStdString(
                    std::to_string(versionPacket.GimbalSerialNumber));
        QString firmwareVersion = QString::fromStdString(
                    std::to_string(versionPacket.GimbalFirmwareVersionMajor)+"."+
                    std::to_string(versionPacket.GimbalFirmwareVersionMinor)+"."+
                    std::to_string(versionPacket.GimbalFirmwareVersionRevision)+"."+
                    std::to_string(versionPacket.GimbalFirmwareVersionBuild));
        QString hardwareVersion = QString::fromStdString(
                    std::to_string(versionPacket.GimbalHardwareVersionMajor)+"."+
                    std::to_string(versionPacket.GimbalHardwareVersionMinor));
        QString protocolVersion = QString::fromStdString(
                    std::to_string(versionPacket.GimbalProtocolVersionMajor)+"."+
                    std::to_string(versionPacket.GimbalProtocolVersionMinor));
        qDebug()<< "SerialNumber:" <<m_context->m_gimbalSerialNumber;
        qDebug()<< "FirmwareVersion:" << m_context->m_firmwareVersion;
        qDebug()<< "HardwareVersion:" << m_context->m_hardwareVersion;
        qDebug()<< "ProtocolVersion:" << m_context->m_protocolVersion;
        m_context->setVersion(gimbalSerialNumber,
                              firmwareVersion,
                              hardwareVersion,
                              protocolVersion);
        if(!m_isGimbalConnected)
        {
            //first time we have found the gimbal so configure the gimbal
            PacketSetup();
        }
        m_isGimbalConnected = true;

    }
}

void CM160Gimbal::ParseCombinedPositionVelocityState(GimbalPacket packet){
    //qDebug("Parsing PTZ Packet");
    UavvCurrentGimbalPositionRate combinedPanTiltPositionVelocityPacket;
    if (UavvCurrentGimbalPositionRate::TryParse(packet,&combinedPanTiltPositionVelocityPacket) == ParseResult::Success)
    {
        /**/
        m_context->setGimbalInfo(
                m_context->m_panPosition,
                combinedPanTiltPositionVelocityPacket.PanVelocity,
                m_context->m_tiltPosition,
                combinedPanTiltPositionVelocityPacket.TiltVelocity);
       /*
        FILE *log=fopen("log.txt","a");
       char data[256];
       sprintf(data,"%.05f,%.05f,%.05f,%.05f",
               combinedPanTiltPositionVelocityPacket.PanPosition,
               combinedPanTiltPositionVelocityPacket.TiltPosition,
               combinedPanTiltPositionVelocityPacket.PanVelocity,
               combinedPanTiltPositionVelocityPacket.TiltVelocity
               );
       printf("Current rate: %s\r\n",data);
       fwrite(data,strlen(data),1,log);
       fwrite("\r\n",strlen("\r\n"),1,log);
       fclose(log);
       */
    }
}

void CM160Gimbal::ParseZoomPosition(GimbalPacket packet){
    qDebug("Parsing Zoom Packet");
    UavvZoomPositionResponse zoomPacket;
    ParseResult result = UavvZoomPositionResponse::TryParse(packet,&zoomPacket);

    if (result == ParseResult::Success)
    {
//        m_context->ZoomPosition((float)zoomPacket.ZoomPositionResponse);
    }else{
        if(result== ParseResult::InvalidData){
            //qDebug("InvalidData");
        }else if(result== ParseResult::InvalidId){
            //qDebug("InvalidId");
        }else if(result== ParseResult::InvalidLength){
            //qDebug("InvalidLength");
        }
    }
}

void CM160Gimbal::ParseSensorFOV(GimbalPacket packet){
//    printf("Parse Sensor FOV size: %d\r\n",packet.Data.size());
    UavvSensorCurrentFoV value;
    ParseResult result = UavvSensorCurrentFoV::TryParse(packet,&value);

    if (result == ParseResult::Success)
    {
        for(int i=0 ;i < value.numSensor; i++){
            m_context->setFOVSensor((int)value.Type[i],value.Horizontal[i],value.Vertical[i]);
        }

    }
}

void CM160Gimbal::ParseCurrentGimbalMode(GimbalPacket packet){
    UavvCurrentGimbalMode value;
    if (UavvCurrentGimbalMode::TryParse(packet,&value) == ParseResult::Success)
    {
        //qDebug("Valid Version Packet");
        /**/
        QString mode = m_context->m_lockMode;
        switch ((CurrentGimbalMode)(value.GimbalMode)) {
        case CurrentGimbalMode::Unarmed:
//            mode = "Unarmed";
            mode = "FREE";
            break;
        case CurrentGimbalMode::ObjectTracking:
//            mode = "ObjectTracking";
            mode = "TRACK";
            break;
        case CurrentGimbalMode::RateControl:
//            mode = "RateControl";
            mode = "FREE";
            break;
        case CurrentGimbalMode::PositionControl:
//            mode = "PositionControl";
            mode = "FREE";
            break;
        case CurrentGimbalMode::Stowed:
//            mode = "Stowed";
            mode = "FREE";
            break;
        case CurrentGimbalMode::SceneSteering:
//            mode = "SceneSteering";
            mode = "VISUAL";
            break;
        case CurrentGimbalMode::Arming:
//            mode = "Arming";
//            mode = "LOCK_FREE";
            break;
        case CurrentGimbalMode::Geolocking:
//            mode = "Geolocking";
            mode = "GEO";
            break;
        case CurrentGimbalMode::Reserved:
//            mode = "Reserved";
            break;
        case CurrentGimbalMode::PerformingFFC:
//            mode = "PerformingFFC";
            break;
        }
        m_context->setGimbalMode(mode);
    }
}

void CM160Gimbal::ParseCornerLocations(GimbalPacket packet){
    UavvCurrentCornerLocation value;
    ParseResult result = UavvCurrentCornerLocation::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setCorners(
                    value.TopLeftLatitude,value.TopLeftLongitude,
                    value.TopRightLatitude,value.TopRightLongitude,
                    value.BottomRightLatitude,value.BottomRightLongitude,
                    value.BottomLeftLatitude,value.BottomLeftLongitude,
                    value.CenterLatitude,value.CenterLongitude
                    );

    }
}

void CM160Gimbal::ParseCurrentGeolockLocations(GimbalPacket packet){
    UavvCurrentGeolockSetpoint value;
    ParseResult result = UavvCurrentGeolockSetpoint::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setGeolockLocation(
                    value.Latitude,value.Longtitude,value.Altitude
                    );

    }
}

void CM160Gimbal::ParseCurrentTargetLocations(GimbalPacket packet){
    UavvCurrentTargetLocation value;
    ParseResult result = UavvCurrentTargetLocation::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setCurrentTargetLocation(
            value.Latitude,value.Longitude,value.SlantRange
        );
    }
}

void CM160Gimbal::ParsePositionPacket(GimbalPacket packet){
    UavvPlatformPosition value;
    ParseResult result = UavvPlatformPosition::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setPlatformPosition(
                    value.Latitude,
                    value.Longtitude,
                    value.Altitude);
    }
}

void CM160Gimbal::ParsePlatformPacket(GimbalPacket packet){
    UavvPlatformOrientation value;
    ParseResult result = UavvPlatformOrientation::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setPlatformOrientation(
                    value.EulerRoll,
                    value.EulerPitch,
                    value.EulerYaw);

    }
}

void CM160Gimbal::ParsePlatformOffsetPacket(GimbalPacket packet){
    UavvGimbalOrientationOffset value;
    ParseResult result = UavvGimbalOrientationOffset::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        /*
        printf("Orientation offset Received (roll ,pitch,yaw) = (%f,%f,%f)\r\n",
               value.Roll,
               value.Pitch,
               value.Yaw);

        m_context->setPlatformOrientation(
                    value.Roll,
                    value.Pitch,
                    value.Yaw);
        */
    }
}

void CM160Gimbal::ParseLaserRange(GimbalPacket packet){
//    printf("[%d]============================================\r\n",
//           m_context->m_laserRangeStart);
    LaserRange value;
    ParseResult result = LaserRange::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {

//        if(m_context->m_laserRangeStart == 0)
        {
            float range = (float)value.Range*5000000/(pow(2,32)-1);
            m_context->setLaserRange(range,(int)value.StatusFlag);
        }
//        m_context->m_laserRangeStart ++;



//        printf("Laser Range = %f\r\n",range);
    }
}
void CM160Gimbal::ParseTrackingParams(GimbalPacket packet){
//    printf("ParseTrackingParams\r\n");
    UavvTrackingParameters value;
    ParseResult result = UavvTrackingParameters::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        QString trackMode;
        if(value.getModeTrackingParameters() == TrackingParametersAction::MovingObjects){
            trackMode = "moving";
        }else if(value.getModeTrackingParameters() == TrackingParametersAction::StationaryObject){
            trackMode = "station";
        }else if(value.getModeTrackingParameters() == TrackingParametersAction::NoChange){
            trackMode = m_context->m_trackMode;
        }
        m_context->setTrackParams((int)value.getAcqTrackingParameters(),
                                               trackMode);

//        printf("Laser Range = %f\r\n",);
    }
}
void CM160Gimbal::ParseRecordingStatus(GimbalPacket packet){
    UavvCurrentRecordingState value;
    ParseResult result = UavvCurrentRecordingState::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        bool recording = value.Recording == 0x01;
        m_context->setRecordingStatus(recording);
    }
}
void CM160Gimbal::ParsePrimarySensor(GimbalPacket packet){
    UavvSetPrimaryVideo value;
    ParseResult result = UavvSetPrimaryVideo::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setSensorID((int)value.PrimaryVideoSensor);
    }
}
void CM160Gimbal::ParseEStab(GimbalPacket packet){
    UavvStabilisationParameters value;
    ParseResult result = UavvStabilisationParameters::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setStabDigital(value.Enable==0x01?true:false);
    }
}
void CM160Gimbal::ParseOverlay(GimbalPacket packet){
    UavvOverlay value;
    ParseResult result = UavvOverlay::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
//        m_context->setStabDigital(value.Enable==0x01?true:false);
    }
}
void CM160Gimbal::ParseVideoDestination(GimbalPacket packet){
    UavvVideoDestination value;
    ParseResult result = UavvVideoDestination::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        unsigned int address = value.IPAddress;
        int port = value.Port;
        char ip[256];
        unsigned char bytes[4];
        bytes[0] = address & 0xFF;
        bytes[1] = (address >> 8) & 0xFF;
        bytes[2] = (address >> 16) & 0xFF;
        bytes[3] = (address >> 24) & 0xFF;
        sprintf(ip,"%d.%d.%d.%d",bytes[3],bytes[2],bytes[1],bytes[0]);
        printf("Stream: %s:%d\r\n",ip,port);
        m_context->setVideoDestination(ip,port);
    }
}
void CM160Gimbal::ParseGyroStab(GimbalPacket packet){
    UavvEnableGyroStabilisation value;
    ParseResult result = UavvEnableGyroStabilisation::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        m_context->setStabGimbal(value.PanFlag==0x01,value.TiltFlag==0x01);
    }
}
void CM160Gimbal::ParseStowMode(GimbalPacket packet){
    UavvStowMode value;
    ParseResult result = UavvStowMode::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {
        QString gimbalMode = "NA";
        switch (value.StowMode) {
        case (unsigned char)StowModeType::ExitStow:
            gimbalMode = "ON";
            break;
        case (unsigned char)StowModeType::EnterStow:
            gimbalMode = "OFF";
            break;
        default:
            break;
        }
        m_context->setGimbalMode(gimbalMode);
    }
}
void CM160Gimbal::ParseTrackMode(GimbalPacket packet){
    UavvModifyObjectTrack value;
    ParseResult result = UavvModifyObjectTrack::TryParse(packet,&value);
    if (result == ParseResult::Success)
    {

    }
}
