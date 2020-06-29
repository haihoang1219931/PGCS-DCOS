#include "TreronGimbal.h"
#include "Camera/VideoEngine/VideoEngineInterface.h"
#include "Joystick/JoystickLib/JoystickThreaded.h"

#include "Packet/MotionImage.h"
#include "Packet/SystemStatus.h"
#include "Packet/TrackResponse.h"
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
#include "TreronGimbalPacketParser.h"

TreronGimbal::TreronGimbal(GimbalInterface *parent) : GimbalInterface(parent)
{
    m_buffer = new BufferOut();
    _receiveSocket = new QUdpSocket();
    _packetParser = new TreronGimbalPacketParser();
    _ipcCommands = new IPCCommands();
    _systemCommands = new SystemCommands();
    _motionCCommands = new MotionCCommands();
    _geoCommands = new GeoCommands();
}
void TreronGimbal::connectToGimbal(Config* config){
    if(config == nullptr)
        return;
    m_config = config;
    m_context->m_hfovMin[0] = 2.33f;
    m_context->m_hfovMax[0] = 63.1f;
    m_context->m_zoom[0] = 1;
    m_context->m_recording = false;
    m_context->m_processOnBoard = true;
    m_context->m_videoStabMode = true;
    m_buffer->connectToHost(m_config->value("Settings:GimbalIP:Value:data").toString(),
                            m_config->value("Settings:GimbalPortIn:Value:data").toInt());
    _receiveSocket->bind(QHostAddress::AnyIPv4,
        static_cast<quint16>(m_config->value("Settings:GimbalPortOut:Value:data").toInt()));

//    connect(_receiveSocket, SIGNAL(readyRead()), this,
//                SLOT(handlePacketReceived()));
//    connect(_packetParser, &TreronGimbalPacketParser::gimbalPacketParsed,
//            this, &TreronGimbal::handlePacketParsed);
    _ipcCommands->m_buffer = m_buffer;
    _systemCommands->m_buffer = m_buffer;
    _motionCCommands->m_buffer = m_buffer;
    _geoCommands->m_buffer = m_buffer;
    _ipcCommands->m_gimbalModel = m_context;
    _systemCommands->m_gimbalModel = m_context;
    _motionCCommands->m_gimbalModel = m_context;
    _geoCommands->m_gimbalModel = m_context;
    /**/
    connect(m_context, SIGNAL(NotifyPropertyChanged(QString)), this,
            SLOT(changeGimbalInfo(QString)));
    m_isGimbalConnected = true;
}
void TreronGimbal::disconnectGimbal(){
    m_buffer->uinit();
    _receiveSocket->close();
    m_isGimbalConnected = false;
}
void TreronGimbal::discoverOnLan(){

}
void TreronGimbal::setPanRate(float rate){

}
void TreronGimbal::setTiltRate(float rate){

}
void TreronGimbal::setGimbalRate(float panRate,float tiltRate){

}
void TreronGimbal::setPanPos(float pos){

}
void TreronGimbal::setTiltPos(float pos){

}
void TreronGimbal::setGimbalPos(float panPos,float tiltPos){

}
void TreronGimbal::setEOZoom(QString command, float value){
//    Q_UNUSED(command);


}
void TreronGimbal::setIRZoom(QString command){
    Q_UNUSED(command);
}
void TreronGimbal::changeSensor(QString sensorID){
    _ipcCommands->changeSensorID(sensorID);
    if(sensorID == "IR"){
        m_context->setSensorID(1);
    }else{
        m_context->setSensorID(0);
    }
    Q_EMIT digitalZoomMaxChanged();
    Q_EMIT zoomMaxChanged();
    Q_EMIT zoomMinChanged();
    Q_EMIT zoomChanged();
}
void TreronGimbal::snapShot(){

}
void TreronGimbal::setGimbalMode(QString mode){
    Q_UNUSED(mode);
}
void TreronGimbal::setGimbalPreset(QString mode){
    Q_UNUSED(mode);
}
void TreronGimbal::setGimbalRecorder(bool enable){
    Q_UNUSED(enable);
}
void TreronGimbal::changeTrackSize(float trackSize){
    m_context->m_trackSize = static_cast<int>(trackSize);
}
void TreronGimbal::setDigitalStab(bool enable){
    _ipcCommands->enableImageStab(enable?"ISTAB_ON":"ISTAB_OFF",enable?0.2:0.0);
    m_context->m_videoStabMode = enable;
}
void TreronGimbal::setRecord(bool enable){
    m_context->m_recording = !m_context->m_recording;
    _ipcCommands->changeRecordMode(!m_context->m_recording?"RECORD_OFF":"RECORD_FULL",0,0);

}
void TreronGimbal::setShare(bool enable){
    m_context->m_gcsShare = !m_context->m_gcsShare;
    if(m_videoEngine != nullptr){
        m_videoEngine->setShare(m_context->m_gcsShare);
    }
}
void TreronGimbal::setLockMode(QString mode, QPointF location){
    m_context->m_lockMode = mode;
//    _ipcCommands->changeLockMode("LOCK_"+mode, "GEOLOCATION_OFF");
    if(mode == "VISUAL"){
//        _ipcCommands->doSceneSteering(0);
        _ipcCommands->setClickPoint(0,location.x()*1920,location.y()*1080,1920,1080,
                                    static_cast<double>(100),
                                    static_cast<double>(100));
    }else if(mode == "FREE"){
        _ipcCommands->changeLockMode("LOCK_FREE", "GEOLOCATION_OFF");

    }else if(mode == "TRACK"){
        _ipcCommands->changeLockMode("LOCK_TRACK", "GEOLOCATION_OFF");
        _ipcCommands->setClickPoint(0,
                                    location.x()*1920,
                                    location.y()*1080,
                                    1920,1080,
                                    static_cast<double>(100),
                                    static_cast<double>(100));
    }
}
void TreronGimbal::setGeoLockPosition(QPoint location){
    Q_UNUSED(location);
}
void TreronGimbal::handlePacketReceived()
{
    while (_receiveSocket->hasPendingDatagrams()) {
        QHostAddress ip = QHostAddress::Any; // any ipaddress
        quint16 port = 0;
        unsigned char receivedData[1024];
        // qint64 received =
        // _receiveSocket->read((char*)receivedData,sizeof(receivedData));
        qint64 received = _receiveSocket->readDatagram(
                              (char *)receivedData, sizeof(receivedData), &ip, &port);
//        printf("%s [%d]bytes\r\n",__func__,received);
        if (received > 0) {
            _packetParser->Push(receivedData, received);
            _packetParser->Parse();
        }
    }

    // qDebug()<<"Reading data done";
}
void TreronGimbal::handlePacketParsed(key_type _key, vector<byte> _data)
{
//    printf("\nKey: %d\n Data: ", _key);
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
        //            m_context->updateGPSData(tele.getPn(),
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
        m_context->setCorners(
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
        m_context->setGeolockLocation(
            geolockPoint.getPn(), geolockPoint.getPe(), geolockPoint.getPd());
        break;
    }

    case (key_type)EyePhoenixProtocol::MotionCStatus: {
        Eye::MotionCStatus motionCState;
        motionCState.parse(_data);
        // TODO: Compare received status with current status. If this state is
        // different from current state, through an ERROR
//        m_context->updateMotionCStatus(
//            motionCState.getPanStabMode() == (byte)Status::StabMode::ON &&
//            motionCState.getTiltStabMode() == (byte)Status::StabMode::ON,
//            motionCState.getPanPos(), motionCState.getTiltPos(),
//            motionCState.getPanVelo(), motionCState.getTiltVelo());
        break;
    }

    case (key_type)EyePhoenixProtocol::SystemStatus: {
        Eye::SystemStatus systemState;
        systemState.parse(_data);
        //            printf("systemState.getIndex() =
        //            %d\r\n",systemState.getIndex());
        Eye::MotionCStatus motionCStatus = systemState.getMotionCStatus();
//        m_context->updateMotionCStatus(motionCStatus);
        Eye::IPCStatusResponse ipcStatusResponse = systemState.getIPCStatus();
        // TODO: Compare received status with current status. If this state is
        // different from current state, through an ERROR
//        printf("\nTrackSize = %f", ipcStatusResponse.getTrackSize());
//        m_context->updateIPCState(ipcStatusResponse);
//        m_context->m_rbSystem->add(systemState);
        Telemetry tele = systemState.getTelemetry();
        //            printf("\nRecevied gps data = [%f, %f, %f]", tele.getPn(),
        //            tele.getPe(), tele.getPd());
//        m_context->updateGPSData(
//            tele.getPn(), tele.getPe(), tele.getPd(), tele.getRoll() / PI * 180.0f,
//            tele.getPitch() / PI * 180.0f, tele.getYaw() / PI * 180.0f);
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
//        printf("Delta: Steering Point: [%d]-[%f, %f, %f, %f]\r\n",
//               xPoint.getIndex(), xPoint.getPx(), xPoint.getPy(),
//               xPoint.getWidth(), xPoint.getHeight());
//        m_context->m_rbXPointEO->add(xPoint);
        break;
    }

    case (key_type)EyePhoenixProtocol::EOTrackingResponse: {
//        Eye::TrackResponse trackRes;
//        trackRes.parse(_data);
        //        printf("\nDelta: Track Point: [%d]-[%f, %f, %f, %f, %f, %f]",
        //               trackRes.getIndex(), trackRes.getPx(), trackRes.getPy(),
        //               trackRes.getWidth(), trackRes.getHeight(),
        //               trackRes.getObjWidth(), trackRes.getObjHeight());
//        m_context->m_rbTrackResEO->add(trackRes);
        break;
    }

    case (key_type)EyePhoenixProtocol::IRSteeringResponse: {
        Eye::XPoint xPoint;
        xPoint.parse(_data);
        //            printf("\nDelta: Steering Point: [%d]-[%f, %f, %f, %f]",
        //            xPoint.getIndex(), xPoint.getPx(), xPoint.getPy(),
        //            xPoint.getWidth(), xPoint.getHeight());
//        m_context->m_rbXPointIR->add(xPoint);
        break;
    }

    case (key_type)EyePhoenixProtocol::IRTrackingResponse: {
//        Eye::TrackResponse trackRes;
//        trackRes.parse(_data);
        //            printf("\nDelta: Track Point: [%d]-[%f, %f, %f, %f, %f,
        //            %f]",trackRes.getIndex(), trackRes.getPx(), trackRes.getPy(),
        //            trackRes.getWidth(), trackRes.getHeight(),
        //            trackRes.getObjWidth(), trackRes.getObjHeight());
//        m_context->m_rbTrackResIR->add(trackRes);
        break;
    }

    case (key_type)EyePhoenixProtocol::GLValid: {
        GPSData gpsData;
        gpsData.parse(_data);
//        m_context->updateGLMeasured(gpsData.getPn(), gpsData.getPe(),
//                gpsData.getPd());
//        gimbalInfoChanged("GLValid");
        break;
    }

    case (key_type)EyePhoenixProtocol::GLInvalid: {
        GPSData gpsData;
        gpsData.parse(_data);
//        m_context->updateGLMeasured(gpsData.getPn(), gpsData.getPe(),
//                gpsData.getPd());
//        gimbalInfoChanged("GLInvalid");
        break;
    }

    case (key_type)EyePhoenixProtocol::EOMotionDataResponse: {
//        Eye::MotionImage motionData;
//        motionData.parse(_data);
//        m_context->m_rbIPCEO->add(motionData);
        break;
    }

    case (key_type)EyePhoenixProtocol::IRMotionDataResponse: {
//        Eye::MotionImage motionData;
//        motionData.parse(_data);
//        m_context->m_rbIPCIR->add(motionData);
        break;
    }

    default: {
        break;
    }
    }
}
