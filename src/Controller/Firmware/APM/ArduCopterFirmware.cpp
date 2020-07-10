#include "ArduCopterFirmware.h"
#include "../../Vehicle/Vehicle.h"
ArduCopterFirmware::ArduCopterFirmware(Vehicle* vehicle)
{
    m_vehicle = vehicle;
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

    m_mapFlightModeOnAir.insert(STABILIZE, "Stabilize");
    m_mapFlightModeOnAir.insert(LOITER,    "Loiter");
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
void ArduCopterFirmware::sendHomePosition(Vehicle* vehicle,QGeoCoordinate location){
    mavlink_home_position_t homePosition;
    homePosition.latitude = static_cast<int32_t>(location.latitude()*pow(10,7));
    homePosition.longitude = static_cast<int32_t>(location.latitude()*pow(10,7));
    mavlink_message_t msg;
    mavlink_msg_home_position_encode_chan(
                static_cast<uint8_t>(2),
                static_cast<uint8_t>(vehicle->communication()->componentId()),
                vehicle->communication()->mavlinkChannel(),
                &msg,
                &homePosition);
    printf("ArduCopterFirmware %s pos(%f,%f)\r\n",__func__,location.latitude(),location.longitude());
    vehicle->sendMessageOnLink(vehicle->communication(), msg);
}
void ArduCopterFirmware::initializeVehicle(Vehicle *vehicle)
{
    vehicle->requestDataStream(MAV_DATA_STREAM_RAW_SENSORS,     2);
    vehicle->requestDataStream(MAV_DATA_STREAM_EXTENDED_STATUS, 2);
    vehicle->requestDataStream(MAV_DATA_STREAM_RC_CHANNELS,     2);
//    vehicle->requestDataStream(MAV_DATA_STREAM_RAW_CONTROLLER,  10);
    vehicle->requestDataStream(MAV_DATA_STREAM_POSITION,        5);//position
    vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA1,          6);//attitude
    vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA2,          6);//attitude
    vehicle->requestDataStream(MAV_DATA_STREAM_EXTRA3,          2);//sensor
}
QString ArduCopterFirmware::gotoFlightMode(void) const
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
void ArduCopterFirmware::commandRTL(void)
{
}
void ArduCopterFirmware::commandLand(void)
{
}

void ArduCopterFirmware::commandTakeoff(Vehicle *vehicle, double altitudeRelative)
{
    Q_UNUSED(altitudeRelative);
    double minimumAltitude = minimumTakeoffAltitude();
    double vehicleAltitudeAMSL = vehicle->altitudeAMSL();
    double takeoffAltRel = vehicleAltitudeAMSL > minimumAltitude ?
                           vehicleAltitudeAMSL : minimumAltitude;
    vehicle->sendMavCommand(vehicle->defaultComponentId(),
                            MAV_CMD_NAV_TAKEOFF,
                            true, // show error
                            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                            static_cast<float>(takeoffAltRel)); // Relative altitude
    printf("takeoffAltRel = %f\r\n", takeoffAltRel);
}

double ArduCopterFirmware::minimumTakeoffAltitude(void)
{
    return 10;
}

void ArduCopterFirmware::commandGotoLocation(Vehicle *vehicle,const QGeoCoordinate &gotoCoord)
{
    printf("ArduCopterFirmware %s (%f,%f,%f)\r\n",__func__,
           gotoCoord.latitude(),gotoCoord.longitude(),
           vehicle->altitudeRelative());
    float altitudeSet = static_cast<float>(vehicle->altitudeRelative());
    mavlink_message_t msg;
    mavlink_set_position_target_global_int_t cmd;

    memset(&cmd, 0, sizeof(cmd));

    cmd.target_system    = static_cast<uint8_t>(vehicle->id());
    cmd.target_component = static_cast<uint8_t>(vehicle->defaultComponentId());
    cmd.coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT;
    cmd.type_mask = 0xFFF8; // Only x/y/z valid
    cmd.lat_int = static_cast<int32_t>(gotoCoord.latitude()*1E7);
    cmd.lon_int = static_cast<int32_t>(gotoCoord.longitude()*1E7);
    cmd.alt = static_cast<float>(altitudeSet);
//    cmd.x = 0.0f;
//    cmd.y = 0.0f;
//    cmd.z = static_cast<float>(-newAltitude);

    mavlink_msg_set_position_target_global_int_encode_chan(
        static_cast<uint8_t>(vehicle->communication()->systemId()),
        static_cast<uint8_t>(vehicle->communication()->componentId()),
        vehicle->communication()->mavlinkChannel(),
        &msg,
        &cmd);

    vehicle->sendMessageOnLink(vehicle->communication(), msg);
}

void ArduCopterFirmware::commandChangeAltitude(double altitudeChange)
{
    Q_UNUSED(altitudeChange);

}

void ArduCopterFirmware::commandSetAltitude(Vehicle *vehicle,double newAltitude)
{
    Q_UNUSED(newAltitude);
    float altitudeSet = static_cast<float>(vehicle->homePosition().altitude()
                                           + newAltitude);
    mavlink_message_t msg;
    mavlink_set_position_target_global_int_t cmd;

    memset(&cmd, 0, sizeof(cmd));

    cmd.target_system    = static_cast<uint8_t>(vehicle->id());
    cmd.target_component = static_cast<uint8_t>(vehicle->defaultComponentId());
    cmd.coordinate_frame = MAV_FRAME_GLOBAL_RELATIVE_ALT;
    cmd.type_mask = 0xFFF8; // Only x/y/z valid
    cmd.lat_int = static_cast<int32_t>(vehicle->coordinate().latitude()*1E7);
    cmd.lon_int = static_cast<int32_t>(vehicle->coordinate().longitude()*1E7);
    cmd.alt = static_cast<float>(newAltitude);
//    cmd.x = 0.0f;
//    cmd.y = 0.0f;
//    cmd.z = static_cast<float>(-newAltitude);

    mavlink_msg_set_position_target_global_int_encode_chan(
        static_cast<uint8_t>(vehicle->communication()->systemId()),
        static_cast<uint8_t>(vehicle->communication()->componentId()),
        vehicle->communication()->mavlinkChannel(),
        &msg,
        &cmd);

    vehicle->sendMessageOnLink(vehicle->communication(), msg);
}

void ArduCopterFirmware::commandChangeSpeed(Vehicle* vehicle,double speedChange)
{
    if (vehicle != nullptr) {
        vehicle->params()->_writeParameterRaw(m_airSpeedParamName,speedChange*100/3.6);
    }
}

void ArduCopterFirmware::commandOrbit(const QGeoCoordinate &centerCoord,
        double radius, double amslAltitude)
{
    Q_UNUSED(centerCoord);
    Q_UNUSED(radius);
    Q_UNUSED(amslAltitude);
}

void ArduCopterFirmware::pauseVehicle(void)
{
}

void ArduCopterFirmware::emergencyStop(void)
{
}

void ArduCopterFirmware::abortLanding(double climbOutAltitude)
{
    Q_UNUSED(climbOutAltitude);
}

void ArduCopterFirmware::startMission(Vehicle *vehicle)
{
    vehicle->sendMavCommand(vehicle->defaultComponentId(), MAV_CMD_MISSION_START, true /*show error */);
}

void ArduCopterFirmware::setCurrentMissionSequence(Vehicle *vehicle, int seq)
{
    Q_UNUSED(seq);

    if (vehicle->flightMode() == "RTL") {
        vehicle->setFlightMode("Auto");
    }

    mavlink_message_t msg;
    printf("setCurrentMissionSequence to %d\r\n", seq);
    mavlink_msg_mission_set_current_pack_chan(vehicle->communication()->systemId(),
            vehicle->communication()->componentId(),
            vehicle->communication()->mavlinkChannel(),
            &msg,
            vehicle->id(),
            vehicle->_compID,
            seq);
    vehicle->sendMessageOnLink(vehicle->m_com, msg);
}

void ArduCopterFirmware::rebootVehicle()
{
}

void ArduCopterFirmware::clearMessages()
{
}

void ArduCopterFirmware::triggerCamera(void)
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

void ArduCopterFirmware::motorTest(Vehicle *vehicle, int motor, int percent)
{
    Q_UNUSED(motor);
    Q_UNUSED(percent);
    vehicle->sendMavCommand(vehicle->defaultComponentId(), MAV_CMD_DO_MOTOR_TEST, true, motor, MOTOR_TEST_THROTTLE_PERCENT, percent, 1, 0, MOTOR_TEST_ORDER_BOARD);
}
void ArduCopterFirmware::setHomeHere(Vehicle* vehicle,float lat, float lon, float alt){
    printf("Set home altitude = %f\r\n",static_cast<double>(alt));
    vehicle->sendMavCommand(vehicle->defaultComponentId(), MAV_CMD_DO_SET_HOME, true,
                            0,0,0,0, lat,lon,alt);
}
