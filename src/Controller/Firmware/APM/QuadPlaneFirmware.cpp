#include "QuadPlaneFirmware.h"
#include "../../Vehicle/Vehicle.h"
QuadPlaneFirmware::QuadPlaneFirmware(Vehicle* vehicle)
{
    m_vehicle = vehicle;
    loadFromFile("conf/Properties.conf");
    m_rtlAltParamName = "ALT_HOLD_RTL";
    m_airSpeedParamName = "TRIM_ARSPD_CM";
    m_loiterRadiusParamName = "WP_LOITER_RAD";
    m_mapFlightMode.insert(MANUAL,         "Manual");
    m_mapFlightMode.insert(CIRCLE,         "Circle");
    m_mapFlightMode.insert(STABILIZE,      "Stabilize");
    m_mapFlightMode.insert(TRAINING,       "Training");
    m_mapFlightMode.insert(ACRO,           "Acro");
    m_mapFlightMode.insert(FLY_BY_WIRE_A,  "FBW A");
    m_mapFlightMode.insert(FLY_BY_WIRE_B,  "FBW B");
    m_mapFlightMode.insert(CRUISE,         "Cruise");
    m_mapFlightMode.insert(AUTOTUNE,       "Autotune");
    m_mapFlightMode.insert(AUTO,           "Auto");
    m_mapFlightMode.insert(RTL,            "RTL");
    m_mapFlightMode.insert(LOITER,         "Loiter");
    m_mapFlightMode.insert(GUIDED,         "Guided");
    m_mapFlightMode.insert(INITIALIZING,   "Initializing");
    m_mapFlightMode.insert(QSTABILIZE,     "QuadPlane Stabilize");
    m_mapFlightMode.insert(QHOVER,         "QuadPlane Hover");
    m_mapFlightMode.insert(QLOITER,        "QuadPlane Loiter");
    m_mapFlightMode.insert(QLAND,          "QuadPlane Land");
    m_mapFlightMode.insert(QRTL,           "QuadPlane RTL");

    m_mapFlightModeOnGround.insert(MANUAL,         "Manual");

//    m_mapFlightModeOnAir.insert(MANUAL,         "Manual");
//    m_mapFlightModeOnAir.insert(FLY_BY_WIRE_A,  "FBW A");
//    m_mapFlightModeOnAir.insert(QHOVER,         "QuadPlane Hover");
}
QString QuadPlaneFirmware::flightMode(int flightModeId){
    if(m_mapFlightMode.contains(flightModeId)) {
        return m_mapFlightMode[flightModeId];
    }else
        return "UNDEFINED";
}
bool QuadPlaneFirmware::flightModeID(QString flightMode,int* base_mode,int* custom_mode){
    bool containFlightMode = false;
    for(int i=0; i< m_mapFlightMode.keys().length();i++){
        int tmpKey = m_mapFlightMode.keys().at(i);
        QString tmpMode = m_mapFlightMode.value(tmpKey);
        if(tmpMode == flightMode){
            *custom_mode = tmpKey;
            *base_mode = 81;
            containFlightMode = true;
            break;
        }
    }
    return containFlightMode;
}
void QuadPlaneFirmware::initializeVehicle(){
    if (m_vehicle == nullptr)
        return;
    m_vehicle->requestDataStream(MAV_DATA_STREAM_RAW_SENSORS,     2);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTENDED_STATUS, 2);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_RC_CHANNELS,     2);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_RAW_CONTROLLER,  10);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_POSITION,        3);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA1,          10);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA2,          10);
    m_vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA3,          10);
//    Q_UNUSED(vehicle);
}
QString QuadPlaneFirmware::gotoFlightMode() const
{
    return QStringLiteral("Guided");
}
bool QuadPlaneFirmware::setFlightMode(const QString& flightMode, uint8_t* base_mode, uint32_t* custom_mode){
    Q_UNUSED(flightMode);
    Q_UNUSED(base_mode);
    Q_UNUSED(custom_mode);
    return false;
}

void QuadPlaneFirmware::commandRTL()
{

}
void QuadPlaneFirmware::commandLand(){

}

void QuadPlaneFirmware::commandTakeoff( double altitudeRelative){
    if (m_vehicle == nullptr)
        return;
    printf("QuadPlaneFirmware %s\r\n",__func__);
    Q_UNUSED(altitudeRelative);
    setCurrentMissionSequence(0);
    m_vehicle->setFlightMode("Auto");

}

double QuadPlaneFirmware::minimumTakeoffAltitude(){
    return 10;
}

void QuadPlaneFirmware::commandGotoLocation(const QGeoCoordinate& gotoCoord){
    Q_UNUSED(gotoCoord);
}

void QuadPlaneFirmware::commandSetAltitude(double newAltitude){
    if (m_vehicle == nullptr)
        return;
    printf("%s = %f\r\n",__func__,newAltitude);
// uint16_t mavlink_msg_mission_item_int_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
//    mavlink_message_t* msg,
//        uint8_t target_system,uint8_t target_component,uint16_t seq,uint8_t frame,uint16_t command,uint8_t current,uint8_t autocontinue,float param1,float param2,float param3,float param4,int32_t x,int32_t y,float z,uint8_t mission_type)
    mavlink_message_t   messageOut;
    mavlink_msg_mission_item_int_pack_chan(m_vehicle->communication()->systemId(),
                                           m_vehicle->communication()->componentId(),
                                           m_vehicle->communication()->mavlinkChannel(),
                                           &messageOut,
                                           static_cast<uint8_t>(m_vehicle->id()),
                                           MAV_COMP_ID_AUTOPILOT1,
                                           0,
                                           static_cast<uint8_t>(MAV_FRAME_GLOBAL_RELATIVE_ALT),
                                           static_cast<uint8_t>(16),
                                           3,//missionRequest.seq == 0,
                                           1,//item->autoContinue(),
                                           0,//item->param1(),
                                           0,//item->param2(),
                                           0,//item->param3(),
                                           0,//item->param4(),
                                           0,//item->param5() * qPow(10.0, 7.0),
                                           0,//item->param6() * qPow(10.0, 7.0),
                                           static_cast<float>(newAltitude),//item->param7(),
                                           0//static_cast<uint8_t>(m_planType)
                                           );
    m_vehicle->sendMessageOnLink(m_vehicle->communication(), messageOut);
}

void QuadPlaneFirmware::commandChangeSpeed(double speedChange){
    if (m_vehicle != nullptr) {
        m_vehicle->params()->_writeParameterRaw(m_airSpeedParamName,speedChange*100/3.6);
    }
}

void QuadPlaneFirmware::commandOrbit(const QGeoCoordinate& centerCoord,
                                     double radius, double amslAltitude){
    Q_UNUSED(centerCoord);
    Q_UNUSED(radius);
    Q_UNUSED(amslAltitude);
}

void QuadPlaneFirmware::pauseVehicle(){

}

void QuadPlaneFirmware::emergencyStop(){

}

void QuadPlaneFirmware::abortLanding(double climbOutAltitude){
    Q_UNUSED(climbOutAltitude);
}

void QuadPlaneFirmware::startMission(){
    if (m_vehicle == nullptr)
        return;
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(), MAV_CMD_MISSION_START, true /*show error */);
}

void QuadPlaneFirmware::setCurrentMissionSequence( int seq){
    Q_UNUSED(seq);
    if (m_vehicle == nullptr)
        return;
    if(m_vehicle->flightMode()=="RTL"){
        m_vehicle->setFlightMode("Auto");
    }
    mavlink_message_t msg;
    printf("setCurrentMissionSequence to %d\r\n",seq);
    mavlink_msg_mission_set_current_pack_chan(m_vehicle->communication()->systemId(),
                                              m_vehicle->communication()->componentId(),
                                              m_vehicle->communication()->mavlinkChannel(),
                                              &msg,
                                              m_vehicle->id(),
                                              m_vehicle->_compID,
                                              seq);
    m_vehicle->sendMessageOnLink(m_vehicle->m_com,msg);
}

void QuadPlaneFirmware::rebootVehicle(){

}

void QuadPlaneFirmware::clearMessages(){

}

void QuadPlaneFirmware::triggerCamera(){

}
void QuadPlaneFirmware::sendPlan(QString planFile){
    Q_UNUSED(planFile);
}

/// Used to check if running current version is equal or higher than the one being compared.
//  returns 1 if current > compare, 0 if current == compare, -1 if current < compare

int QuadPlaneFirmware::versionCompare(QString& compare){
    Q_UNUSED(compare);
    return 0;
}
int QuadPlaneFirmware::versionCompare(int major, int minor, int patch){
    Q_UNUSED(major);
    Q_UNUSED(minor);
    Q_UNUSED(patch);
    return 0;
}

void QuadPlaneFirmware::motorTest(int motor, int percent){
    Q_UNUSED(motor);
    Q_UNUSED(percent);
    if (m_vehicle == nullptr)
        return;
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(), MAV_CMD_DO_MOTOR_TEST, true, motor, MOTOR_TEST_THROTTLE_PERCENT, percent, 1, 0, MOTOR_TEST_ORDER_BOARD);

}
void QuadPlaneFirmware::setHomeHere(float lat, float lon, float alt){
    if (m_vehicle == nullptr)
        return;
    m_vehicle->sendMavCommand(m_vehicle->defaultComponentId(), MAV_CMD_DO_SET_HOME, true,
                              0,0,0,0, lat,lon,alt);
}

void QuadPlaneFirmware::setGimbalMode(QString mode)
{
    Q_UNUSED(mode);
}
