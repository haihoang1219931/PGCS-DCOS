#include "GremseyGimbal.h"
#include "GimbalControl.h"
#include "SensorController.h"
#include "Payload/VideoEngine/VideoEngineInterface.h"
#include "Utils/Joystick/JoystickThreaded.h"
#include "Flight/Firmware/FirmwarePlugin.h"
#define rad2Deg 57.2957795f

GremseyGimbal::GremseyGimbal(GimbalInterface *parent) : GimbalInterface(parent)
{
    m_targetLocation = new TargetLocalization();
    m_targetLocation->visionViewInit(0.006f,1920,1080);
    m_zoom.append(0x0000); //1x
    m_zoom.append(0x0DC1); //2x
    m_zoom.append(0x186C); //3x
    m_zoom.append(0x2015); //4x
    m_zoom.append(0x2594); //5x
    m_zoom.append(0x29B7); //6x
    m_zoom.append(0x2CFB); //7x
    m_zoom.append(0x2FB0); //8x
    m_zoom.append(0x320C); //9x
    m_zoom.append(0x342D); //10x
    m_zoom.append(0x3608); //11x
    m_zoom.append(0x37AA); //12x
    m_zoom.append(0x391C); //13x
    m_zoom.append(0x3A66); //14x
    m_zoom.append(0x3B90); //15x
    m_zoom.append(0x3C9C); //16x
    m_zoom.append(0x3D91); //17x
    m_zoom.append(0x3E72); //18x
    m_zoom.append(0x3F40); //19x
    m_zoom.append(0x4000); // 20x * 1
    m_zoom.append(0x6000); // 20x * 2
    m_zoom.append(0x6AAB); // 20x * 3
    m_zoom.append(0x7000); // 20x * 4
    m_zoom.append(0x7334); // 20x * 5
    m_zoom.append(0x7556); // 20x * 6
    m_zoom.append(0x76DC); // 20x * 7
    m_zoom.append(0x7800); // 20x * 8
    m_zoom.append(0x78E4); // 20x * 9
    m_zoom.append(0x799A); // 20x * 10
    m_zoom.append(0x7A2F); // 20x * 11
    m_zoom.append(0x7AC0); // 20x * 12
    for(int i=0; i< m_zoom.size(); i++){
        if(i<20){
            m_mapZoom[m_zoom[i]] = i+1;
        }else{
            m_mapZoom[m_zoom[i]] = 20*(i-20+2);
        }
    }
    m_context->m_hfovMax[0] = 70.2f;
    m_context->m_hfovMax[1] = 17.7f;
    m_context->m_hfovMin[0] = 2.2f;
    m_context->m_hfovMin[1] = 17.7f;
    m_context->m_hfov[0] = m_context->m_hfovMax[0];
    m_context->m_hfov[1] = 17.7f;
    m_context->m_zoom[0] = 1;
    m_context->m_zoom[1] = 1;
    setZoom(0,1.0f);
    setZoomMax(0,20);
    setZoomMin(0,1);
    setZoomTarget(0,1);
    setDigitalZoomMax(0,12);

    setZoom(1,1.0f);
    setZoomMax(1,1);
    setZoomMin(1,1);
    setZoomTarget(1,1);
    setDigitalZoomMax(1,8);

    Q_EMIT zoomCalculatedChanged(0,1);
}
void GremseyGimbal::setJoystick(JoystickThreaded* joystick){
    if(joystick == nullptr) return;
    m_joystick = joystick;
    connect(m_joystick,&JoystickThreaded::axisValueChanged,this,&GremseyGimbal::handleAxisValueChanged);
    connect(m_joystick,&JoystickThreaded::buttonStateChanged,this,&GremseyGimbal::handleButtonStateChanged);
    //    m_context->m_hfov[m_context->m_sensorID] = 62.9f;
}
void GremseyGimbal::handleButtonStateChanged(int buttonID, bool pressed){
    if(m_joystick != nullptr){
        if(buttonID >=0 && buttonID < m_joystick->buttonCount() && pressed){
            QString mapFunc = m_joystick->button(buttonID)->mapFunc();
            printf("button[%d] pressed=%s mapFunc=%s\r\n",
                   buttonID,pressed?"true":"false",mapFunc.toStdString().c_str());
            if(mapFunc == "EO/IR"){
                if(context()->m_sensorID == 0)
                    changeSensor("IR");
                else{
                    changeSensor("EO");
                }
            }else if(mapFunc == "SNAPSHOT"){
                snapShot();
            }else if(mapFunc == "VISUAL" || mapFunc == "FREE"){
                setLockMode(mapFunc);
            }else if(mapFunc.contains("PRESET")){
                QString nextPreset = "OFF";
                if(m_context->m_presetMode.contains("OFF")){
                    nextPreset = "FRONT";
                }else if(m_context->m_presetMode.contains("FRONT")){
                    nextPreset = "RIGHT";
                }else if(m_context->m_presetMode.contains("RIGHT")){
                    nextPreset = "BEHIND";
                }else if(m_context->m_presetMode.contains("BEHIND")){
                    nextPreset = "LEFT";
                }else if(m_context->m_presetMode.contains("LEFT")){
                    nextPreset = "NADIR";
                }else if(m_context->m_presetMode.contains("NADIR")){
                    nextPreset = "OFF";
                }
                setGimbalPreset(nextPreset);
            }else if(mapFunc == "DIGITAL_STAB"){
                setDigitalStab(!m_context->m_videoStabMode);
            }else if(mapFunc == "RECORD"){
                setRecord(!m_context->m_recording);
            }
        }
    }
}

void GremseyGimbal::handleAxisValueChanged(int axisID, float value){
    if(!m_joystick->pic()){
        if(m_joystick->axisCount()<3) return;
        // send raw gimbal rate
        float maxAxis = 32768.0f;
//        float maxRate = 1024.0f/maxAxis;
        float maxRate = 120/maxAxis;
        float deadZone = 160;
        float alphaSpeed = 0.5;
        float invertPan = m_joystick->invertPan();
        float invertTilt = m_joystick->invertTilt();
        float invertZoom = m_joystick->invertZoom();
        float panRate = m_joystick->axis(m_joystick->axisPan())->value();
        float tiltRate = m_joystick->axis(m_joystick->axisTilt())->value();
        float zoomRate = m_joystick->axis(m_joystick->axisZoom())->value();
        if(fabs(panRate) < deadZone) panRate = 0;
        if(fabs(tiltRate) < deadZone) tiltRate = 0;
        if(fabs(zoomRate) < deadZone) zoomRate = 0;
        float x = invertPan * (panRate-deadZone*(panRate>=0?1:-1)) * maxRate;
        float y = invertTilt * (tiltRate-deadZone*(tiltRate>=0?1:-1)) * maxRate;
        float z = invertZoom * zoomRate;
        float hfov = m_context->m_hfov[m_context->m_sensorID];
        float panRateScale = (alphaSpeed * hfov * x / m_context->m_hfovMax[m_context->m_sensorID]);
        float tiltRateScale = (alphaSpeed *  hfov * y / m_context->m_hfovMax[m_context->m_sensorID]);
//                        printf("GREMSY hfov[%.02f] hfovMax[%.02f] x[%.02f] y[%.02f] z[%.02f] panRateScale[%.02f] tiltRateScale[%.02f]\r\n",
//                               hfov,m_context->m_hfovMax[m_context->m_sensorID],
//                                x,y,z,
//                                panRateScale,tiltRateScale);

        if(m_videoEngine == nullptr
                || m_context->m_lockMode == "FREE"){
            setGimbalRate((panRateScale),(tiltRateScale));
            m_videoEngine->moveImage(0,
                                     0,
                                     invertZoom * zoomRate);
        }else{
            m_videoEngine->moveImage(-invertPan * panRate,
                                     -invertTilt * tiltRate,
                                     invertZoom * zoomRate);
        }
        if(m_joystick->axisZoom() == axisID){
            if(m_context->m_sensorID == 0){
                if(z > deadZone)
                    setEOZoom("ZOOM_OUT",0);
                else if(z < -deadZone)
                    setEOZoom("ZOOM_IN",0);
                else
                    setEOZoom("ZOOM_STOP",0);
            }
        }
        if(m_gimbalCurrentMode != "RATE_MODE")
        {
            m_panRateJoystick = 0;
            m_tiltRateJoystick = 0;
            setGimbalPreset("OFF");
        }
    }
}
void GremseyGimbal::connectToGimbal(Config* config){
    if(config == nullptr)
        return;
    m_timerQueryZoom = new QTimer();
    m_timerQueryZoom->setInterval(100);
    m_timerQueryZoom->setSingleShot(false);

    m_config = config;
    printf("Gimbal[%s:%d]\r\n",m_config->value("Settings:GimbalIP:Value:data").toString().toStdString().c_str(),
           m_config->value("Settings:GimbalPortIn:Value:data").toInt());
    printf("Sensor[%s:%d]\r\n",m_config->value("Settings:SensorIP:Value:data").toString().toStdString().c_str(),
           m_config->value("Settings:SensorPortIn:Value:data").toInt());
    m_sensor = new SensorController();
    m_sensor->connectToHost(m_config->value("Settings:SensorIP:Value:data").toString(),
                            m_config->value("Settings:SensorPortIn:Value:data").toInt());
    connect(m_timerQueryZoom,&QTimer::timeout,this,&GremseyGimbal::sendQueryZoom);
    connect(m_sensor,&SensorController::dataReceived,this,&GremseyGimbal::handleQuery);
    m_timerQueryZoom->start();
    setEOZoom("",1);
}
void GremseyGimbal::disconnectGimbal(){

}

void GremseyGimbal::changeSensor(QString sensorID){
    if(m_videoEngine!=nullptr){
        if(sensorID == "IR"){
            m_videoEngine->setVideo(m_config->value("Settings:StreamIR:Value:data").toString());
            m_context->setSensorID(1);
        }else{
            m_videoEngine->setVideo(m_config->value("Settings:StreamEO:Value:data").toString());
            m_context->setSensorID(0);
        }
        Q_EMIT digitalZoomMaxChanged();
        Q_EMIT zoomMaxChanged();
        Q_EMIT zoomMinChanged();
        Q_EMIT zoomChanged();
    }
}
void GremseyGimbal::setSensorColor(QString sensorID,QString colorMode){
    if(m_videoEngine!=nullptr){
        if(sensorID == "IR"){
            m_videoEngine->setSensorColor(colorMode);
        }else{

        }
    }
}
void GremseyGimbal::setEOZoom(QString command, float value){
    if(command == "ZOOM_IN"){
        m_sensor->sendRawData("068101040727FF");
    }else if(command == "ZOOM_OUT"){
        m_sensor->sendRawData("068101040737FF");
    }else if(command == "ZOOM_STOP"){
        m_sensor->sendRawData("068101040700FF");
    }else {
//        printf("GremseyGimbal::%s %s\r\n",__func__,command.toStdString().c_str());
        if(value>=1 && value<= 240){
            int zoomPosition;
            int zoomBegin = value<=20? static_cast<int>(value)-1: static_cast<int>(value/20)-1+19;
            int zoomEnd = zoomBegin +1;
            if(value<=20)
                zoomPosition = m_zoom[zoomBegin] +
                        (value-zoomBegin-1)/(zoomEnd-zoomBegin)*(m_zoom[zoomEnd]-m_zoom[zoomBegin]);
            else{
                zoomPosition = m_zoom[zoomBegin] +
                        (value- ((zoomBegin-19)+1)*20)/((zoomEnd-zoomBegin) * 20.0f)*
                        (m_zoom[zoomEnd]-m_zoom[zoomBegin]);
            }
            char cmd[32];
            sprintf(cmd,"%04X",zoomPosition);

            QString zoomPositionRaw = QString::fromStdString(std::string(cmd));
            QString zoomPositionCommand = QString("0681010447") +
                    QString("0") + zoomPositionRaw[0]+
                    QString("0") + zoomPositionRaw[1]+
                    QString("0") + zoomPositionRaw[2]+
                    QString("0") + zoomPositionRaw[3]+"FF";
            //            printf("zoom[%.02f] equal %s from zoomBegin[%d][%04X] to zoomEnd[%d][%04X] zoomPositionCommand=%s\r\n",
            //                   value,cmd,zoomBegin,m_zoom[zoomBegin],zoomEnd,m_zoom[zoomEnd],
            //                   zoomPositionCommand.toStdString().c_str());
            m_sensor->sendRawData(zoomPositionCommand);
        }
    }
}
void GremseyGimbal::setIRZoom(QString command){
    float zoomDigital = 1;
    if(command == "1x"){
        zoomDigital = 1;
    }else if(command == "2x"){
        zoomDigital = 2;
    }else if(command == "4x"){
        zoomDigital = 4;
    }else if(command == "8x"){
        zoomDigital = 8;
    }
    printf("%s %f\r\n",__func__,zoomDigital);
    m_videoEngine->setdigitalZoom(zoomDigital);
}
void GremseyGimbal::setGimbalRate(float panRate,float tiltRate){
//    if(m_gimbal->getConnectionStatus() == 1){
//        m_gimbal->set_Mode(e_control_gimbal_mode::GIMBAL_LOCK_MODE);
//        m_gimbal->set_control(0,
//                              static_cast<int>(tiltRate),
//                              static_cast<int>(panRate));
//    }
//    printf("GremseyGimbal::%s panRate=[%f] tiltRate=[%f]\r\n",__func__,panRate,tiltRate);
    if (m_vehicle != nullptr) {
        m_vehicle->setGimbalRate(panRate,tiltRate);
    }
}
void GremseyGimbal::snapShot(){
    if(m_videoEngine!=nullptr){
        m_videoEngine->capture();
        struct TargetLocalization::GpsPosition from,to;
        from.Latitude = m_context->m_latitude;
        from.Longitude = m_context->m_longitude;
        to.Latitude = m_context->m_centerLat;
        to.Longitude = m_context->m_centerLon;
        float flatDistance = m_targetLocation->distanceFlatEarth(from,to);
        QGeoCoordinate target(m_context->m_centerLat,m_context->m_centerLon);
        Q_EMIT functionHandled("SNAPSHOT",target,flatDistance);
    }
}
void GremseyGimbal::changeTrackSize(float trackSize){
    if(m_videoEngine!=nullptr){
        m_videoEngine->changeTrackSize((int)trackSize);
    }
}
void GremseyGimbal::setDigitalStab(bool enable){
    if(m_videoEngine!=nullptr){
        m_videoEngine->setStab(enable);
    }
    if(m_context != nullptr){
        m_context->m_videoStabMode = enable;
    }
}
void GremseyGimbal::setLockMode(QString mode, QPointF location){
    m_context->m_lockMode = mode;
    if(mode == "FREE"){
        setDigitalStab(false);
    }else if(mode == "TRACK"){
        m_videoEngine->setTrackAt(0,
                                  m_videoEngine->sourceSize().width()/2,
                                  m_videoEngine->sourceSize().height()/2,
                                  m_videoEngine->sourceSize().width(),
                                  m_videoEngine->sourceSize().height());
        resetTrackParam();
        setDigitalStab(true);
    }else if(mode == "VISUAL"){
        setDigitalStab(true);
    }
}
void GremseyGimbal::setRecord(bool enable){
    m_context->m_recording = !m_context->m_recording;
    if(m_videoEngine != nullptr){
        m_videoEngine->setRecord(m_context->m_recording);
    }
}
void GremseyGimbal::setShare(bool enable){
    m_context->m_gcsShare = !m_context->m_gcsShare;
    if(m_videoEngine != nullptr){
        m_videoEngine->setShare(m_context->m_gcsShare);
    }
}

void GremseyGimbal::setGimbalPreset(QString mode)
{
    if(m_vehicle != nullptr)
    {
        if(mode == "NEXT"){
            QString nextPreset = "OFF";
            if(m_context->m_presetMode.contains("OFF")){
                nextPreset = "FRONT";
            }else if(m_context->m_presetMode.contains("FRONT")){
                nextPreset = "RIGHT";
            }else if(m_context->m_presetMode.contains("RIGHT")){
                nextPreset = "BEHIND";
            }else if(m_context->m_presetMode.contains("BEHIND")){
                nextPreset = "LEFT";
            }else if(m_context->m_presetMode.contains("LEFT")){
                nextPreset = "NADIR";
            }else if(m_context->m_presetMode.contains("NADIR")){
                nextPreset = "OFF";
            }
            m_context->m_presetMode = nextPreset;
        }else{
            m_context->m_presetMode = mode;
        }

        printf("%s from %s to %s\r\n",__func__,
               m_context->m_presetMode.toStdString().c_str(),
               mode.toStdString().c_str());
        if (mode.contains("OFF")) {
            setGimbalMode("RATE_MODE");
            return;
        }
//        if(m_gimbalCurrentMode == "ANGLE_BODY_MODE")
//            handleGimbalModeChanged("ANGLE_BODY_MODE");
//        else
            setGimbalMode("ANGLE_BODY_MODE");
    }else{
        printf("m_vehicle == nullptr\r\n");
    }
}

void GremseyGimbal::setGimbalMode(QString mode)
{
    if(m_vehicle != nullptr &&  m_vehicle->m_firmwarePlugin != nullptr)
    {
        m_vehicle->m_firmwarePlugin->setGimbalMode(mode);
         printf("set preset mode \r\n");
    }
}

void GremseyGimbal::setGimbalPos(float panPos, float tiltPos)
{
    if(m_vehicle != nullptr &&  m_vehicle->m_firmwarePlugin != nullptr)
    {
        m_vehicle->m_firmwarePlugin->setGimbalAngle(panPos,tiltPos);
        printf("set pan pos:%5.1f\r\n",-panPos);
    }
}
void GremseyGimbal::setObjectSearch(bool enable){
    if(m_videoEngine!= nullptr){
        m_context->m_gcsSearch = enable;
        m_videoEngine->setObjectSearch(enable);
    }
}
void GremseyGimbal::setVehicle(Vehicle *vehicle)
{
    m_vehicle = vehicle;
    if(m_vehicle!= nullptr)
    {
        connect(m_vehicle,SIGNAL(mavlinkGimbalMessageReceived(mavlink_message_t)),this,SLOT(handleGimbalMessage(mavlink_message_t)));
        connect(m_vehicle,SIGNAL(mavlinkMessageReceived(mavlink_message_t)),this,SLOT(handleVehicleMessage(mavlink_message_t)));
        connect(m_vehicle,SIGNAL(linkChanged()),this,SLOT(handleVehicleLinkChanged()));
        if(m_vehicle->m_firmwarePlugin !=nullptr)
        {
        connect(m_vehicle,SIGNAL(gimbalModeChanged(QString)),this,SLOT(handleGimbalModeChanged(QString)));
        connect(m_vehicle,SIGNAL(gimbalModeSetFail()),this,SLOT(handleGimbalSetModeFail()));
        }
    }
}
void GremseyGimbal::sendQueryZoom(){
    m_sensor->sendRawData("0681090447FF");
}
void GremseyGimbal::handleQuery(QString data){
    if(data.startsWith("9050") && data.size() == 14 && data.endsWith("FF"))
    {
        // zoom position
        QString zoomPosition = QString(data[5])+QString(data[7])+QString(data[9])+QString(data[11]);
        int zoomPos = zoomPosition.toInt(nullptr,16);
        float zoomPosF = static_cast<float>(zoomPos);
        for(int i=0; i< m_zoom.size()-1; i++){
            if(zoomPos >= m_zoom[i] && zoomPos <= m_zoom[i+1]){
                zoomPosF = static_cast<float>(m_mapZoom[m_zoom[i]]) +
                        static_cast<float>(zoomPos - m_zoom[i])/static_cast<float>(m_zoom[i+1]-m_zoom[i]) *
                        (static_cast<float>(m_mapZoom[m_zoom[i+1]]) - static_cast<float>(m_mapZoom[m_zoom[i]]));

                m_context->m_zoom[0] = zoomPosF;
                m_context->m_hfov[0] = atanf(tan(m_context->m_hfovMax[0]/2/180*M_PI)/zoomPosF)/M_PI*180*2;
                Q_EMIT GimbalInterface::zoomChanged();
                break;
            }
        }
    }
}

void GremseyGimbal::handleGimbalMessage(mavlink_message_t message)
{
    switch (message.msgid) {
    case MAVLINK_MSG_ID_HEARTBEAT:
        break;
    case MAVLINK_MSG_ID_MOUNT_ORIENTATION:
        mavlink_mount_orientation_t mount_orient;
        mavlink_msg_mount_orientation_decode(&message, &mount_orient);

        m_context->m_panPosition = mount_orient.yaw;
        m_context->m_tiltPosition = mount_orient.pitch;
//        printf("pan gimbal: %5.1f\r\n",static_cast<double>(m_context->m_panPosition));
//        printf("tilt gimbal: %5.1f\r\n",static_cast<double>(m_context->m_tiltPosition));
//        printf("altitude: %5.1f\r\n",static_cast<double>(m_context->m_altitudeOffset));
        break;
    default:
        break;
    }
}

void GremseyGimbal::handleVehicleMessage(mavlink_message_t message)
{
    // uav: lat,long,alt, roll, pitch, yaw
    switch (message.msgid) {
    case MAVLINK_MSG_ID_HEARTBEAT:
        break;

    case MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
        mavlink_global_position_int_t gps_pos;
        mavlink_msg_global_position_int_decode(&message,&gps_pos);
        m_context->m_latitude = (static_cast<float>(gps_pos.lat))/10000000;
        m_context->m_longitude = (static_cast<float>(gps_pos.lon))/10000000;
        m_context->m_altitudeOffset = m_vehicle->altitudeAGL();
        if(m_targetLocation!=nullptr){            
            double center[2];
            double corner[4][2];
            double uavPosition[2];
            uavPosition[0] = m_context->m_latitude;
            uavPosition[1] = m_context->m_longitude;
            m_targetLocation->visionViewMain(
                m_context->m_hfov[m_context->m_sensorID] / rad2Deg,
                m_context->m_rollOffset / rad2Deg,
                m_context->m_pitchOffset / rad2Deg,
                m_context->m_yawOffset / rad2Deg,
                m_context->m_panPosition / rad2Deg,
                m_context->m_tiltPosition / rad2Deg,
                uavPosition,
                m_context->m_altitudeOffset,
                0,
                center,corner[0],corner[1],corner[3],corner[2]
            );
            m_context->m_centerLat = center[0];
            m_context->m_centerLon = center[1];

            m_context->m_cornerLat[0] = corner[0][0];
            m_context->m_cornerLon[0] = corner[0][1];
            m_context->m_cornerLat[1] = corner[1][0];
            m_context->m_cornerLon[1] = corner[1][1];
            m_context->m_cornerLat[2] = corner[2][0];
            m_context->m_cornerLon[2] = corner[2][1];
            m_context->m_cornerLat[3] = corner[3][0];
            m_context->m_cornerLon[3] = corner[3][1];
        }
        break;

    case MAVLINK_MSG_ID_ATTITUDE:
        mavlink_attitude_t  attitude;
        mavlink_msg_attitude_decode(&message,&attitude);
        m_context->m_rollOffset  = attitude.roll * rad2Deg;
        m_context->m_pitchOffset = attitude.pitch * rad2Deg;
        m_context->m_yawOffset   = attitude.yaw * rad2Deg;
        //printf("att-roll: %5.1f\r\n",m_context->m_rollOffset);
        break;
    default:
        break;
    }
}

void GremseyGimbal::handleVehicleLinkChanged()
{
    if(m_vehicle != nullptr)
    {
        if(m_vehicle->link() == true && m_vehicle->m_firmwarePlugin != nullptr)
        {
            //set gimbal mode
            m_vehicle->m_firmwarePlugin->setGimbalMode("RATE_MODE");
        }
    }
}

void GremseyGimbal::handleGimbalModeChanged(QString mode)
{
    m_gimbalCurrentMode = mode;
    if(m_context->m_presetMode.contains("OFF") && mode == "RATE_MODE")
    {
        setGimbalRate(m_panRateJoystick,m_tiltRateJoystick);
        Q_EMIT presetChanged(true);
        return;
    }
    if(mode == "ANGLE_BODY_MODE")
    {
        if(m_context->m_presetMode.contains("FRONT")){
            m_presetAngleSet_Pan = 0;
            m_presetAngleSet_Tilt = 0;
        }else if(m_context->m_presetMode.contains("RIGHT")){
//            m_presetAngleSet_Pan = 90;
            m_presetAngleSet_Pan = abs(m_context->m_panPosition - 90) < 180 ? 90 : -270;
            m_presetAngleSet_Tilt = 0;
        }else if(m_context->m_presetMode.contains("LEFT")){
//            m_presetAngleSet_Pan = -90;
            m_presetAngleSet_Pan = abs(m_context->m_panPosition + 90) < 180 ? -90 : 270;
            m_presetAngleSet_Tilt = 0;
        }else if(m_context->m_presetMode.contains("BEHIND")){
//            m_presetAngleSet_Pan = -180;
            m_presetAngleSet_Pan = abs(m_context->m_panPosition + 180) < 180 ? -180 : 179;
            m_presetAngleSet_Tilt = 0;
        }else if(m_context->m_presetMode.contains("NADIR")){
            m_presetAngleSet_Pan = 0;
            m_presetAngleSet_Tilt = 90;
        }
        setGimbalPos(-m_presetAngleSet_Pan,-m_presetAngleSet_Tilt);
        Q_EMIT presetChanged(true);

    }
}

void GremseyGimbal::handleGimbalSetModeFail()
{
    Q_EMIT presetChanged(false);
}
void GremseyGimbal::enableDigitalZoom(bool enable){

}
