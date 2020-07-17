#include "ArduCopterFirmware.h"
ArduCopterFirmware::ArduCopterFirmware(Vehicle* vehicle)
{
    m_vehicle = vehicle;
    connect(m_vehicle->joystick(),&JoystickThreaded::buttonStateChanged,this,&ArduCopterFirmware::handleJSButton);
    connect(m_vehicle->joystick(),&JoystickThreaded::useJoystickChanged,this,&ArduCopterFirmware::handleUseJoystick);

    //connect(m_vehicle,&Vehicle::mavCommandResult,this,&ArduCopterFirmware::handleMavCommandResult);

    loadFromFile("conf/Properties.conf");
    m_rtlAltParamName = "RTL_ALT";
    m_airSpeedParamName = "WPNAV_SPEED";
    m_mapFlightMode.insert(STABILIZE, "Stabilize");
    m_mapFlightMode.insert(ACRO,      "Acro");
    m_mapFlightMode.insert(ALT_HOLD,  "Altitude Hold");
    m_mapFlightMode.insert(AUTO,      "Auto");
    m_mapFlightMode.insert(GUIDED,    "Guided");
    m_mapFlightMode.insert(LOITER,    "Loiter");
    m_mapFlightMode.insert(RTL,       "RTL");
    m_mapFlightMode.insert(CIRCLE,    "Circle");
    m_mapFlightMode.insert(LAND,      "Land");
    m_mapFlightMode.insert(DRIFT,     "Drift");
    m_mapFlightMode.insert(SPORT,     "Sport");
    m_mapFlightMode.insert(FLIP,      "Flip");
    m_mapFlightMode.insert(AUTOTUNE,  "Autotune");
    m_mapFlightMode.insert(POS_HOLD,  "Position Hold");
    m_mapFlightMode.insert(BRAKE,     "Brake");
    m_mapFlightMode.insert(THROW,     "Throw");
    m_mapFlightMode.insert(AVOID_ADSB, "Avoid ADSB");
    m_mapFlightMode.insert(GUIDED_NOGPS, "Guided No GPS");
    m_mapFlightMode.insert(SAFE_RTL, "Smart RTL");

    m_mapFlightModeOnGround.insert(STABILIZE, "Stabilize");
    m_mapFlightModeOnGround.insert(LOITER,    "Loiter");
    m_mapFlightModeOnAir.insert(STABILIZE, "Stabilize");
    m_mapFlightModeOnAir.insert(LOITER,    "Loiter");
    m_joystickTimer.setInterval(50);
    m_joystickTimer.setSingleShot(false);
    m_joystickClearRCTimer.setInterval(20);
    m_joystickClearRCTimer.setSingleShot(false);
    connect(&m_joystickTimer,&QTimer::timeout,this,&ArduCopterFirmware::sendJoystickData);
    connect(&m_joystickClearRCTimer,&QTimer::timeout,this,&ArduCopterFirmware::sendClearRC);
    m_joystickTimer.start();
    if(m_vehicle->joystick()!=nullptr){
        m_vehicle->setFlightMode(m_vehicle->joystick()->pic()?"Loiter":"Guided");
    }

    m_gimbalHearbeatTimer.setInterval(1000);
    m_gimbalHearbeatTimer.setSingleShot(false);
    connect(&m_gimbalHearbeatTimer,&QTimer::timeout,this,&ArduCopterFirmware::sendGimbalHeartbeat);
    m_gimbalHearbeatTimer.start();
}
ArduCopterFirmware::~ArduCopterFirmware(){
    if(m_joystickTimer.isActive()){
        m_joystickTimer.stop();
    }
}
bool ArduCopterFirmware::pic(){
    return m_pic;
}
QString ArduCopterFirmware::flightMode(int flightModeId)
{
    if (m_mapFlightMode.contains(flightModeId)) {
        return m_mapFlightMode[flightModeId];
    } else
        return "UNDEFINED";
}
bool ArduCopterFirmware::flightModeID(QString flightMode, int *base_mode, int *custom_mode)
{
    bool containFlightMode = false;

    for (int i = 0; i < m_mapFlightMode.keys().length(); i++) {
        int tmpKey = m_mapFlightMode.keys().at(i);
        QString tmpMode = m_mapFlightMode.value(tmpKey);

        if (tmpMode == flightMode) {
            *custom_mode = tmpKey;
            *base_mode = 81;
            containFlightMode = true;
            break;
        }
    }

    return containFlightMode;
}
void ArduCopterFirmware::sendHomePosition(QGeoCoordinate location){
    if (m_vehicle == nullptr)
        return;
    mavlink_home_position_t homePosition;
    homePosition.latitude = static_cast<int32_t>(location.latitude()*pow(10,7));
    homePosition.longitude = static_cast<int32_t>(location.latitude()*pow(10,7));
    mavlink_message_t msg;
    mavlink_msg_home_position_encode_chan(
                static_cast<uint8_t>(2),
                static_cast<uint8_t>(m_vehicle->communication()->componentId()),
                m_vehicle->communication()->mavlinkChannel(),
                &msg,
                &homePosition);
    printf("ArduCopterFirmware %s pos(%f,%f)\r\n",__func__,location.latitude(),location.longitude());
    m_vehicle->sendMessageOnLink(m_vehicle->communication(), msg);
}
void ArduCopterFirmware::initializeVehicle()
{
    if (m_vehicle == nullptr)
        return;
    m_vehicle->requestDataStream(MAV_DATA_STREAM_RAW_SENSORS,     2);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTENDED_STATUS, 2);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_RC_CHANNELS,     2);
    //    m_vehicle->requestDataStream(MAV_DATA_STREAM_RAW_CONTROLLER,  10);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_POSITION,        5);//position
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA1,          6);//attitude
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA2,          6);//attitude
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA3,          2);//sensor

    //set gimbal mode
//    setGimbalMode(CTRL_ANGULAR_RATE);
//    setGimbalMove(m_gimbalLastPan,m_gimbalLastTilt);
}
QString ArduCopterFirmware::gotoFlightMode() const
{
    return QStringLiteral("Guided");
}
bool ArduCopterFirmware::setFlightMode(const QString &flightMode, uint8_t *base_mode, uint32_t *custom_mode)
{
    Q_UNUSED(flightMode);
    Q_UNUSED(base_mode);
    Q_UNUSED(custom_mode);
    return false;
}
void ArduCopterFirmware::commandRTL()
{
}
void ArduCopterFirmware::commandLand()
{
}

void ArduCopterFirmware::commandTakeoff( double altitudeRelative)
{
    if (m_vehicle == nullptr)
        return;
    double minimumAltitude = minimumTakeoffAltitude();
    double vehicleAltitudeAMSL = m_vehicle->altitudeAMSL();
    double takeoffAltRel = altitudeRelative > minimumAltitude ?
                altitudeRelative : minimumAltitude;
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(),
                              MAV_CMD_NAV_TAKEOFF,
                              true, // show error
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                              static_cast<float>(takeoffAltRel)); // Relative altitude
    printf("takeoffAltRel = %f\r\n", takeoffAltRel);
}

double ArduCopterFirmware::minimumTakeoffAltitude()
{
    return 10;
}

void ArduCopterFirmware::commandGotoLocation(const QGeoCoordinate &gotoCoord)
{
    if (m_vehicle == nullptr)
        return;
    printf("ArduCopterFirmware %s (%f,%f,%f)\r\n",__func__,
           gotoCoord.latitude(),gotoCoord.longitude(),
           m_vehicle->altitudeRelative());
    float altitudeSet = static_cast<float>(m_vehicle->altitudeRelative());
    mavlink_message_t msg;
    mavlink_set_position_target_global_int_t cmd;

    memset(&cmd, 0, sizeof(cmd));

    cmd.target_system    = static_cast<uint8_t>(m_vehicle->id());
    cmd.target_component = static_cast<uint8_t>(m_vehicle->defaultComponentId());
    cmd.coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT;
    cmd.type_mask = 0xFFF8; // Only x/y/z valid
    cmd.lat_int = static_cast<int32_t>(gotoCoord.latitude()*1E7);
    cmd.lon_int = static_cast<int32_t>(gotoCoord.longitude()*1E7);
    cmd.alt = static_cast<float>(altitudeSet);
    //    cmd.x = 0.0f;
    //    cmd.y = 0.0f;
    //    cmd.z = static_cast<float>(-newAltitude);

    mavlink_msg_set_position_target_global_int_encode_chan(
                static_cast<uint8_t>(m_vehicle->communication()->systemId()),
                static_cast<uint8_t>(m_vehicle->communication()->componentId()),
                m_vehicle->communication()->mavlinkChannel(),
                &msg,
                &cmd);

    m_vehicle->sendMessageOnLink(m_vehicle->communication(), msg);
}

void ArduCopterFirmware::commandChangeAltitude(double altitudeChange)
{
    Q_UNUSED(altitudeChange);
}

void ArduCopterFirmware::commandSetAltitude(double newAltitude)
{
    if (m_vehicle == nullptr)
        return;
    Q_UNUSED(newAltitude);
    float altitudeSet = static_cast<float>(m_vehicle->homePosition().altitude()
                                           + newAltitude);
    mavlink_message_t msg;
    mavlink_set_position_target_global_int_t cmd;

    memset(&cmd, 0, sizeof(cmd));

    cmd.target_system    = static_cast<uint8_t>(m_vehicle->id());
    cmd.target_component = static_cast<uint8_t>(m_vehicle->defaultComponentId());
    cmd.coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT;
    cmd.type_mask = 0xFFF8; // Only x/y/z valid
    cmd.lat_int = static_cast<int32_t>(m_vehicle->coordinate().latitude()*1E7);
    cmd.lon_int = static_cast<int32_t>(m_vehicle->coordinate().longitude()*1E7);
    cmd.alt = static_cast<float>(newAltitude);
    //    cmd.x = 0.0f;
    //    cmd.y = 0.0f;
    //    cmd.z = static_cast<float>(-newAltitude);

    mavlink_msg_set_position_target_global_int_encode_chan(
                static_cast<uint8_t>(m_vehicle->communication()->systemId()),
                static_cast<uint8_t>(m_vehicle->communication()->componentId()),
                m_vehicle->communication()->mavlinkChannel(),
                &msg,
                &cmd);

    m_vehicle->sendMessageOnLink(m_vehicle->communication(), msg);
}

void ArduCopterFirmware::commandChangeSpeed(double speedChange)
{
    if (m_vehicle != nullptr) {
        m_vehicle->params()->_writeParameterRaw(m_airSpeedParamName,speedChange*100/3.6);
    }
}

void ArduCopterFirmware::commandOrbit(const QGeoCoordinate &centerCoord,
                                      double radius, double amslAltitude)
{
    Q_UNUSED(centerCoord);
    Q_UNUSED(radius);
    Q_UNUSED(amslAltitude);
}

void ArduCopterFirmware::pauseVehicle()
{
}

void ArduCopterFirmware::emergencyStop()
{
}

void ArduCopterFirmware::abortLanding(double climbOutAltitude)
{
    Q_UNUSED(climbOutAltitude);
}

void ArduCopterFirmware::startMission()
{
    if (m_vehicle == nullptr)
        return;
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(), MAV_CMD_MISSION_START, true /*show error */);
}

void ArduCopterFirmware::setCurrentMissionSequence( int seq)
{
    Q_UNUSED(seq);
    if (m_vehicle == nullptr)
        return;
    if (m_vehicle->flightMode() == "RTL") {
        m_vehicle->setFlightMode("Auto");
    }

    mavlink_message_t msg;
    printf("setCurrentMissionSequence to %d\r\n", seq);
    mavlink_msg_mission_set_current_pack_chan(m_vehicle->communication()->systemId(),
                                              m_vehicle->communication()->componentId(),
                                              m_vehicle->communication()->mavlinkChannel(),
                                              &msg,
                                              m_vehicle->id(),
                                              m_vehicle->_compID,
                                              seq);
    m_vehicle->sendMessageOnLink(m_vehicle->m_com, msg);
}

void ArduCopterFirmware::rebootVehicle()
{
}

void ArduCopterFirmware::clearMessages()
{
}

void ArduCopterFirmware::triggerCamera()
{
}
void ArduCopterFirmware::sendPlan(QString planFile)
{
    Q_UNUSED(planFile);
}

/// Used to check if running current version is equal or higher than the one being compared.
//  returns 1 if current > compare, 0 if current == compare, -1 if current < compare

int ArduCopterFirmware::versionCompare(QString &compare)
{
    Q_UNUSED(compare);
    return 0;
}
int ArduCopterFirmware::versionCompare(int major, int minor, int patch)
{
    Q_UNUSED(major);
    Q_UNUSED(minor);
    Q_UNUSED(patch);
    return 0;
}

void ArduCopterFirmware::motorTest( int motor, int percent)
{
    if (m_vehicle == nullptr)
        return;
    Q_UNUSED(motor);
    Q_UNUSED(percent);
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(), MAV_CMD_DO_MOTOR_TEST, true, motor, MOTOR_TEST_THROTTLE_PERCENT, percent, 1, 0, MOTOR_TEST_ORDER_BOARD);
}
void ArduCopterFirmware::setHomeHere(float lat, float lon, float alt){
    if (m_vehicle == nullptr)
        return;
    printf("Set home altitude = %f\r\n",static_cast<double>(alt));
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(), MAV_CMD_DO_SET_HOME, true,
                              0,0,0,0, lat,lon,alt);
}

///control gimbal with mavlink
void ArduCopterFirmware::setGimbalAngle(float pan, float tilt)
{
    if(m_gimbalCurrentMode == "ANGLE_BODY_MODE")
    {
        setGimbalMove(-pan,tilt);
        setGimbalMove(-pan,tilt);
        setGimbalMove(-pan,tilt);
        setGimbalMove(-pan,tilt);
    }
}

void ArduCopterFirmware::setGimbalRate(float pan, float tilt)
{

    if(m_gimbalCurrentMode == "RATE_MODE")
    {        
//        printf("ArduCopterFirmware::%s panRate=[%f] tiltRate=[%f]\r\n",__func__,pan,tilt);
        setGimbalMove(-pan,tilt);    
        m_gimbalLastPan = -pan;
        m_gimbalLastTilt = tilt;
    }
}

void ArduCopterFirmware::setGimbalMode(QString mode)
{
    if(mode == "ANGLE_BODY_MODE"){
        setGimbalMode(CTRL_ANGLE_ABSOLUTE_FRAME, CTRL_ANGLE_ABSOLUTE_FRAME, CTRL_ANGLE_BODY_FRAME);
    }
    else if(mode == "ANGLE_ABSOLUTE_MODE")
    {
        setGimbalMode(CTRL_ANGLE_ABSOLUTE_FRAME, CTRL_ANGLE_ABSOLUTE_FRAME, CTRL_ANGLE_ABSOLUTE_FRAME);
    }
    else if(mode == "RATE_MODE"){
        setGimbalMode(CTRL_ANGULAR_RATE, CTRL_ANGULAR_RATE, CTRL_ANGULAR_RATE);
        setGimbalMove(m_gimbalLastPan,m_gimbalLastTilt);
    }
    m_gimbalSetMode = mode;
}

void ArduCopterFirmware::setGimbalMove(float pan, float tilt)
{
//    printf("%s - pan: %f , tilt: %f\r\n",__func__,pan,tilt);
//    float mode_mavlink_targeting = static_cast<float>(MAV_MOUNT_MODE_MAVLINK_TARGETING);
//    m_vehicle->sendMavCommand(m_vehicle->gimbalComponentId(),MAV_CMD_DO_MOUNT_CONTROL,
//                              false,tilt,0,-pan,0,0,0,mode_mavlink_targeting);
    mavlink_message_t msg;
    mavlink_command_long_t  cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.target_system =     4;
    cmd.target_component =  MAV_COMP_ID_GIMBAL;
    cmd.command =           MAV_CMD_DO_MOUNT_CONTROL;
    cmd.confirmation =      true;
    cmd.param1 =            static_cast<float>(tilt);
    cmd.param2 =            static_cast<float>(0);
    cmd.param3 =            static_cast<float>(-pan);
    cmd.param7 =            static_cast<float>(MAV_MOUNT_MODE_MAVLINK_TARGETING);
//    mavlink_msg_command_long_encode_chan(cmd.target_system ,
//                                         cmd.target_component,
//                                         m_vehicle->communication()->mavlinkChannel(),
//                                         &msg,
//                                         &cmd);
    mavlink_msg_command_long_encode(cmd.target_system ,
                                         MAV_COMP_ID_SYSTEM_CONTROL,
                                         &msg,
                                         &cmd);
    m_vehicle->sendMessageOnLink(m_vehicle->communication(),msg);
}

void ArduCopterFirmware::setGimbalMode(ArduCopterFirmware::control_gimbal_axis_input_mode_t rollMode,
                                       ArduCopterFirmware::control_gimbal_axis_input_mode_t tiltMode,
                                       ArduCopterFirmware::control_gimbal_axis_input_mode_t panMode)
{
    m_vehicle->sendMavCommand(m_vehicle->gimbalComponentId(),MAV_CMD_DO_MOUNT_CONFIGURE,
                              false,MAV_MOUNT_MODE_MAVLINK_TARGETING,1,1,1,rollMode,tiltMode,panMode);

//    mavlink_message_t msg;
//    mavlink_command_long_t  cmd;
//    memset(&cmd, 0, sizeof(cmd));
//    cmd.target_system =     0;
//    cmd.target_component =  154;
//    cmd.command =           MAV_CMD_DO_MOUNT_CONFIGURE;
//    cmd.confirmation =      0;
//    cmd.param1 =            MAV_MOUNT_MODE_MAVLINK_TARGETING;
//    cmd.param2 =            1;
//    cmd.param3 =            1;
//    cmd.param4 =            1;
//    cmd.param5 =            mode;
//    cmd.param6 =            mode;
//    cmd.param7 =            mode;
//    mavlink_msg_command_long_encode_chan(cmd.target_system ,
//                                         cmd.target_component,
//                                         m_vehicle->communication()->mavlinkChannel(),
//                                         &msg,
//                                         &cmd);
//    m_vehicle->sendMessageOnLink(m_vehicle->communication(),msg);
}

void ArduCopterFirmware::sendGimbalHeartbeat()
{
    mavlink_heartbeat_t hearbeat;
    hearbeat.type = MAV_TYPE_ONBOARD_CONTROLLER;
    hearbeat.autopilot = MAV_AUTOPILOT_GENERIC;
    hearbeat.base_mode = 0;
    hearbeat.custom_mode = 0;
    hearbeat.system_status = MAV_STATE_ACTIVE;

    mavlink_message_t message;
    mavlink_msg_heartbeat_encode(m_vehicle->gimbalSystemId(),MAV_COMP_ID_SYSTEM_CONTROL,&message,&hearbeat);

    m_vehicle->sendMessageOnLink(m_vehicle->m_com, message);

}

void ArduCopterFirmware::changeGimbalCurrentMode()
{
    m_gimbalCurrentMode = m_gimbalSetMode;
}

QString ArduCopterFirmware::getGimbalCurrentMode()
{
    return m_gimbalCurrentMode;
}

///end control gimbal

void ArduCopterFirmware::sendJoystickData(){
    if (m_vehicle == nullptr)
        return;

    if(m_vehicle->joystick()== nullptr){
        return;
    }
    if(m_vehicle->joystick()->pic()){

    }else{
        if(m_vehicle->flightMode() == "Loiter"){
            m_vehicle->setFlightMode("Guided");
        }
    }
    if(m_vehicle->joystick()->axisCount() < 4 || !m_vehicle->joystick()->useJoystick()){
        return;
    }
    mavlink_message_t msg;
    JSAxis *axisRoll = m_vehicle->joystick()->axis(m_vehicle->joystick()->axisRoll());
    JSAxis *axitPitch = m_vehicle->joystick()->axis(m_vehicle->joystick()->axisPitch());
    JSAxis *axisYaw = m_vehicle->joystick()->axis(m_vehicle->joystick()->axisYaw());
    JSAxis *axisThrottle = m_vehicle->joystick()->axis(m_vehicle->joystick()->axisThrottle());

    float roll = axisRoll != nullptr?axisRoll->value()*(axisRoll->inverted()?-1:1):0;
    float pitch = axitPitch != nullptr?axitPitch->value()*(axitPitch->inverted()?-1:1):0;
    float yaw = axisYaw != nullptr?axisYaw->value()*(axisYaw->inverted()?-1:1):0;
    float throttle = axisThrottle != nullptr?axisThrottle->value()*(axisThrottle->inverted()?-1:1):0;
    mavlink_msg_rc_channels_override_pack_chan(
                m_vehicle->communication()->systemId(),
                m_vehicle->communication()->componentId(),
                m_vehicle->communication()->mavlinkChannel(),
                &msg,
                m_vehicle->id(),
                m_vehicle->_compID,
                static_cast<uint16_t>(convertRC(m_vehicle->joystick()->pic()?roll:0,1)),
                static_cast<uint16_t>(convertRC(m_vehicle->joystick()->pic()?pitch:0,2)),
                static_cast<uint16_t>(convertRC(m_vehicle->joystick()->pic()?throttle:0,3)),
                static_cast<uint16_t>(convertRC(m_vehicle->joystick()->pic()?yaw:0,4)),
                0,//static_cast<uint16_t>(convertRC(0,5)),
                0,//static_cast<uint16_t>(convertRC(0,6)),
                0,//static_cast<uint16_t>(convertRC(0,7)),
                0,//static_cast<uint16_t>(convertRC(0,8)),
                0,//static_cast<uint16_t>(convertRC(0,9)),
                0,//static_cast<uint16_t>(convertRC(0,10)),
                0,//static_cast<uint16_t>(convertRC(0,11)),
                0,//static_cast<uint16_t>(convertRC(0,12)),
                0,//static_cast<uint16_t>(convertRC(0,13)),
                0,//static_cast<uint16_t>(convertRC(0,14)),
                0,//static_cast<uint16_t>(convertRC(0,15)),
                0,//static_cast<uint16_t>(convertRC(0,16)),
                0,//static_cast<uint16_t>(convertRC(0,17)),
                0//static_cast<uint16_t>(convertRC(0,18))
                );
    m_vehicle->sendMessageOnLink(m_vehicle->communication(),msg);
}
void ArduCopterFirmware::sendClearRC(){
    if(m_sendClearRCCount == m_sendClearRCMax){
        m_joystickClearRCTimer.stop();
    }else{
        printf("%s\r\n",__func__);
        mavlink_message_t msg;
        mavlink_msg_rc_channels_override_pack_chan(
                    m_vehicle->communication()->systemId(),
                    m_vehicle->communication()->componentId(),
                    m_vehicle->communication()->mavlinkChannel(),
                    &msg,
                    m_vehicle->id(),
                    m_vehicle->_compID,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    );
        m_vehicle->sendMessageOnLink(m_vehicle->communication(),msg);
    }
    m_sendClearRCCount++;
}
void ArduCopterFirmware::handleJSButton(int id, bool clicked){
    if(m_vehicle != nullptr && m_vehicle->joystick() != nullptr){
        if(id>=0 && id < m_vehicle->joystick()->buttonCount()){

            JSButton* button = m_vehicle->joystick()->button(id);
            printf("%s[%id] %s\r\n",__func__,id,clicked?"true":"false");
            if(m_mapFlightMode.values().contains(button->mapFunc())){
                if(m_vehicle->joystick()->pic()){
                    if(button->mapFunc() == "RTL")
                        m_vehicle->setFlightMode(button->mapFunc());
                }else{
                    m_vehicle->setFlightMode(button->mapFunc());
                }
            }else if(button->mapFunc() == "PIC/CIC" || button->mapFunc() == "CIC/PIC"){
                m_vehicle->setFlightMode((clicked)?"Loiter":"Guided");
            }
        }
    }
}
void ArduCopterFirmware::handleUseJoystick(bool enable) {
    if(enable){        
        connect(&m_joystickTimer,&QTimer::timeout,this,&ArduCopterFirmware::sendJoystickData);
        m_joystickTimer.start();
    }else{        
        disconnect(&m_joystickTimer,&QTimer::timeout,this,&ArduCopterFirmware::sendJoystickData);
        m_joystickTimer.stop();
        m_sendClearRCCount = 0;
        m_joystickClearRCTimer.start();
    }
}

void ArduCopterFirmware::handleMavCommandResult(int vehicleId, int component, int command, int result, bool noReponseFromVehicle)
{

}

float ArduCopterFirmware::convertRC(float input, int channel){
    float result = 0;
    if(m_vehicle!=nullptr){
        float axisMin = -32768;
        float axisMax = 32768;
        float axisZero = 0;
        float min = m_vehicle->paramsController()->containKey("RC"+QString::fromStdString(std::to_string(channel))+"_MIN")?
                    m_vehicle->paramsController()->getParam("RC"+QString::fromStdString(std::to_string(channel))+"_MIN").toFloat():1000;
        float max = m_vehicle->paramsController()->containKey("RC"+QString::fromStdString(std::to_string(channel))+"_MIN")?
                    m_vehicle->paramsController()->getParam("RC"+QString::fromStdString(std::to_string(channel))+"_MAX").toFloat():2000;
        float trim = m_vehicle->paramsController()->containKey("RC"+QString::fromStdString(std::to_string(channel))+"_MIN")?
                    m_vehicle->paramsController()->getParam("RC"+QString::fromStdString(std::to_string(channel))+"_TRIM").toFloat():1500;

        if(channel == 3){
            trim = (max+min)/2;
        }
        if(input < 0){
            result = (input-axisZero) / (axisMin-axisZero) * (min-trim)+trim;
        }else {
            result = (input-axisZero) / (axisMax-axisZero) * (max-trim)+trim;
        }
//        printf("RC%d[%f - %f - %f] from %f to %f\r\n",channel,min,trim,max,input,result);
    }

    return result;
}
