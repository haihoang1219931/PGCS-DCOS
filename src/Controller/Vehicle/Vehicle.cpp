#include "Vehicle.h"

#include "../Firmware/FirmwarePluginManager.h"
#include "../Firmware/FirmwarePlugin.h"
#include "src/Joystick/JoystickLib/JoystickThreaded.h"
Vehicle::Vehicle(QObject *parent) : QObject(parent)
{

    _defaultComponentId = MAV_COMP_ID_ALL;
    m_uas = new UAS();
    m_firmwarePluginManager = new FirmwarePluginManager();
    m_firmwarePlugin = m_firmwarePluginManager->firmwarePluginForAutopilot(this,_firmwareType, static_cast<MAV_TYPE>(_vehicleType));
    _loadDefaultParamsShow();
    Q_EMIT flightModesChanged();
    Q_EMIT flightModesOnAirChanged();
    _mavHeartbeat.setInterval(1000);
    _mavHeartbeat.setSingleShot(false);
    connect(&_mavHeartbeat, SIGNAL(timeout()), this, SLOT(_sendGCSHeartbeat()));
    _cameraLink.setInterval(5000);
    _cameraLink.setSingleShot(false);
    connect(&_cameraLink, SIGNAL(timeout()), this, SLOT(_checkCameraLink()));
    _mavCommandAckTimer.setSingleShot(true);
    _mavCommandAckTimer.setInterval(_highLatencyLink ? _mavCommandAckTimeoutMSecsHighLatency : _mavCommandAckTimeoutMSecs);
    connect(&_mavCommandAckTimer, &QTimer::timeout, this, &Vehicle::_sendMavCommandAgain);
    m_paramsController = new ParamsController(this);
    //    m_paramsController->_vehicle = this;
}
Vehicle::~Vehicle()
{
}
Vehicle* Vehicle::uav(){
    return m_uav;
}
void Vehicle::setUav(Vehicle* uav){
    if(uav != nullptr){
        m_uav = uav;
    }
}
JoystickThreaded* Vehicle::joystick(){
    return m_joystick;
}
void Vehicle::setJoystick(JoystickThreaded* joystick){
    m_joystick = joystick;
    _pic = m_joystick->pic();
    Q_EMIT picChanged();
    _useJoystick = m_joystick->useJoystick();
    Q_EMIT useJoystickChanged(_useJoystick);
    connect(m_joystick,&JoystickThreaded::picChanged,this,&Vehicle::handlePIC);
    connect(m_joystick,&JoystickThreaded::useJoystickChanged,this,&Vehicle::handleUseJoystick);
}
void Vehicle::handlePIC(){
    _pic = m_joystick->pic();
    Q_EMIT picChanged();
    printf("%s = %s\r\n",__func__,_pic?"true":"false");
}
void Vehicle::handleUseJoystick(bool useJoystick){
    _useJoystick = useJoystick;
    Q_EMIT useJoystickChanged(_useJoystick);
}

ParamsController *Vehicle::paramsController()
{
    return m_paramsController;
}
void Vehicle::setParamsController(ParamsController *paramsController)
{
    m_paramsController = paramsController;
}
PlanController *Vehicle::planController()
{
    return m_planController;
}
void Vehicle::setPlanController(PlanController *planController)
{
    m_planController = planController;
}
IOFlightController *Vehicle::communication()
{
    return m_com;
}
void Vehicle::setCommunication(IOFlightController *com)
{

    _requestPlanAfterParams = false;
    m_com = com;
    _logFile = m_com->getLogFile();
    // Request firmware version
    connect(m_com,SIGNAL(messageReceived(mavlink_message_t)),
            this,SLOT(_mavlinkMessageReceived(mavlink_message_t)));
    connect(this, SIGNAL(_sendMessageOnLinkOnThread(IOFlightController*,mavlink_message_t)),
            this, SLOT(_sendMessageOnLink(IOFlightController*,mavlink_message_t)),Qt::QueuedConnection);
    connect(m_com, SIGNAL(mavlinkMessageStatus(int,uint64_t,uint64_t,uint64_t,float)),
            this, SLOT(_mavlinkMessageStatus(int,uint64_t,uint64_t,uint64_t,float)));
    connect(m_uas, SIGNAL(messagesChanged()),this,SIGNAL(uasChanged()));
    sendMavCommand(MAV_COMP_ID_ALL,                     // Don't know default component id yet.
                   MAV_CMD_REQUEST_PROTOCOL_VERSION,
                   false,                               // No error shown if fails
                   1);                                  // Request protocol version
    // Ask the vehicle for firmware version info.
    sendMavCommand(MAV_COMP_ID_ALL,                         // Don't know default component id yet.
                   MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
                   false,                                   // No error shown if fails
                   1);
    connect(m_paramsController, SIGNAL(missingParametersChanged(bool)),
            this, SIGNAL(missingParametersChanged(bool)));
    connect(m_paramsController, SIGNAL(loadProgressChanged(float)),
            this, SIGNAL(loadProgressChanged(float)));
    connect(m_paramsController, SIGNAL(missingParametersChanged(bool)),
            this, SLOT(_startPlanRequest(void)));

    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->initializeVehicle(this);

    _mavHeartbeat.start();
    _cameraLink.start();
}
ParamsController* Vehicle::params(){
    return m_paramsController;
}
void Vehicle::commandLoiterRadius(float radius){
    m_paramsController->_writeParameterRaw("WP_LOITER_RAD",QVariant::fromValue(radius));
}
void Vehicle::commandRTL(void)
{
    if (m_firmwarePlugin != nullptr) {
        m_firmwarePlugin->commandRTL();
    }
}
void Vehicle::commandLand(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandLand();
}

void Vehicle::commandTakeoff(double altitudeRelative)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandTakeoff(this, altitudeRelative);
}

double Vehicle::minimumTakeoffAltitude(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->minimumTakeoffAltitude();
}

void Vehicle::commandGotoLocation(const QGeoCoordinate &gotoCoord)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandGotoLocation(this,gotoCoord);
}

void Vehicle::commandChangeAltitude(double altitudeChange)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandChangeAltitude(altitudeChange);
}

void Vehicle::commandSetAltitude(double newAltitude)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandSetAltitude(this,newAltitude);
}

void Vehicle::commandChangeSpeed(double speedChange)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandChangeSpeed(this,speedChange);
}

void Vehicle::commandOrbit(const QGeoCoordinate &centerCoord, double radius, double amslAltitude)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->commandOrbit(centerCoord, radius, amslAltitude);
}

void Vehicle::pauseVehicle(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->pauseVehicle();
}

void Vehicle::emergencyStop(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->emergencyStop();
}

void Vehicle::abortLanding(double climbOutAltitude)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->abortLanding(climbOutAltitude);
}

void Vehicle::startMission(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->startMission(this);
}

void Vehicle::startEngine(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->startEngine(this);
}

void Vehicle::setCurrentMissionSequence(int seq)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->setCurrentMissionSequence(this, seq);
}

void Vehicle::rebootVehicle()
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->rebootVehicle();
}

void Vehicle::clearMessages()
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->clearMessages();
}

void Vehicle::triggerCamera(void)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->triggerCamera();
}
void Vehicle::sendPlan(QString planFile)
{
    if (m_firmwarePlugin != nullptr)
        m_firmwarePlugin->sendPlan(planFile);
}

/// Used to check if running current version is equal or higher than the one being compared.
//  returns 1 if current > compare, 0 if current == compare, -1 if current < compare
void Vehicle::setFirmwareVersion(int majorVersion, int minorVersion, int patchVersion, FIRMWARE_VERSION_TYPE versionType)
{
    _firmwareMajorVersion = majorVersion;
    _firmwareMinorVersion = minorVersion;
    _firmwarePatchVersion = patchVersion;
    _firmwareVersionType = versionType;
    Q_EMIT firmwareVersionChanged();
#ifdef DEBUG
    printf("%s %d.%d.%d %s\r\n", __func__, majorVersion, minorVersion, patchVersion,
           firmwareVersionTypeString().toStdString().c_str());
#endif
}

void Vehicle::setFirmwareCustomVersion(int majorVersion, int minorVersion, int patchVersion)
{
    _firmwareCustomMajorVersion = majorVersion;
    _firmwareCustomMinorVersion = minorVersion;
    _firmwareCustomPatchVersion = patchVersion;
    Q_EMIT firmwareCustomVersionChanged();
#ifdef DEBUG
    printf("%s %d.%d.%d %s\r\n", __func__, majorVersion, minorVersion, patchVersion,
           firmwareVersionTypeString().toStdString().c_str());
#endif
}

QString Vehicle::firmwareVersionTypeString(void) const
{
    switch (_firmwareVersionType) {
    case FIRMWARE_VERSION_TYPE_DEV:
        return QStringLiteral("dev");

    case FIRMWARE_VERSION_TYPE_ALPHA:
        return QStringLiteral("alpha");

    case FIRMWARE_VERSION_TYPE_BETA:
        return QStringLiteral("beta");

    case FIRMWARE_VERSION_TYPE_RC:
        return QStringLiteral("rc");

    case FIRMWARE_VERSION_TYPE_OFFICIAL:
    default:
        return QStringLiteral("");
    }
}
int Vehicle::versionCompare(QString &compare)
{
    if (m_firmwarePlugin != nullptr) {
        return m_firmwarePlugin->versionCompare(compare);
    } else
        return -1;
}
int Vehicle::versionCompare(int major, int minor, int patch)
{
    if (m_firmwarePlugin != nullptr) {
        return m_firmwarePlugin->versionCompare(major, minor, patch);
    } else
        return -1;
}
void Vehicle::_mavlinkMessageStatus(int uasId, uint64_t totalSent, uint64_t totalReceived, uint64_t totalLoss, float lossPercent)
{
    if(uasId == _id) {
        _mavlinkSentCount       = totalSent;
        _mavlinkReceivedCount   = totalReceived;
        _mavlinkLossCount       = totalLoss;
        _mavlinkLossPercent     = lossPercent;
        Q_EMIT mavlinkStatusChanged();
    }
}
void Vehicle::motorTest(int motor, int percent)
{
    if (m_firmwarePlugin != nullptr) {
        m_firmwarePlugin->motorTest(this, motor, percent);
    }
}
void Vehicle::setHomeLocation(float lat, float lon){
    if(m_firmwarePlugin != nullptr){
        if(m_planController->m_missionItems.size() > 0)
            m_firmwarePlugin->setHomeHere(this, lat, lon, _altitudeAMSL - _altitudeAGL);
    }
}
void Vehicle::setAltitudeRTL(float alt){
    if(m_firmwarePlugin != nullptr){
        if (m_paramsController != nullptr) {
            if(static_cast<int>(alt) != static_cast<int>(_homeAltitude))
                m_paramsController->_writeParameterRaw(m_firmwarePlugin->rtlAltParamName(),QVariant::fromValue(alt*100));
        }
    }
}
void Vehicle::sendHomePosition(QGeoCoordinate location){
    if(!isnan(location.latitude()) && !isnan(location.longitude())){
        mavlink_home_position_t homePosition;
        homePosition.latitude = static_cast<int32_t>(location.latitude()*pow(10,7));
        homePosition.longitude = static_cast<int32_t>(location.longitude()*pow(10,7));
        mavlink_message_t msg;
        mavlink_msg_home_position_encode_chan(
                    255,
                    190,
                    0,
                    &msg,
                    &homePosition);
        uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
        int len = mavlink_msg_to_send_buffer(buffer, &msg);
        m_com->getInterface()->writeBytesSafe((char*)buffer,len);
    }
}
void Vehicle::activeProperty(QString name,bool active){
    int propertiesShowCount = 0;
    for(int i=0; i< _propertiesModel.size(); i++){
        if(_propertiesModel[i]->name() == name){
            _propertiesModel[i]->setSelected(active);
        }
        if(_propertiesModel[i]->selected())
            propertiesShowCount ++;
    }
    setPropertiesShowCount(propertiesShowCount);
    if(m_firmwarePlugin != nullptr){
        m_firmwarePlugin->saveToFile(QString("conf/Properties.conf"),_propertiesModel);
    }
}
int Vehicle::countActiveProperties(){
    int propertiesShowCount = 0;
    for(int i=0; i< _propertiesModel.size(); i++){
        if(_propertiesModel[i]->selected())
            propertiesShowCount ++;
    }
    printf("%s propertiesShowCount = %d\r\n",__func__,propertiesShowCount);
    setPropertiesShowCount(propertiesShowCount);
}

void Vehicle::_loadDefaultParamsShow(){
    printf("%s\r\n",__func__);
    if(m_firmwarePlugin != nullptr){
        printf("%s m_firmwarePlugin != nullptr\r\n",__func__);
        _propertiesModel.clear();
        _propertiesModel.append(m_firmwarePlugin->listParamsShow());
        Q_EMIT propertiesModelChanged();
        countActiveProperties();
    }else{
        printf("%s m_firmwarePlugin == nullptr\r\n",__func__);
    }
}
void Vehicle::_setPropertyValue(QString name,QString value,QString unit){
    if(name == ""){
        return;
    }
    bool paramExist = false;
    int propertiesShowCount = 0;
    for(int i=0; i< _propertiesModel.size(); i++){
        if(_propertiesModel[i]->name() == name){
            paramExist = true;
            _propertiesModel[i]->setValue(value);
            _propertiesModel[i]->setUnit(unit);
        }
        if(_propertiesModel[i]->selected())
            propertiesShowCount ++;
    }
    if(!paramExist){
        Fact* fact = new Fact(nullptr);
        fact->setSelected(false);
        fact->setName(name);
        fact->setUnit(unit);
        fact->setLowerValue(0);
        fact->setUpperValue(0);
        fact->setLowerColor("transparent");
        fact->setMiddleColor("transparent");
        fact->setUpperColor("transparent");
        _propertiesModel.append(fact);
        Q_EMIT propertiesModelChanged();
    }else{
    }
    setPropertiesShowCount(propertiesShowCount);
}
void Vehicle::_setParamValue(QString name,QString value,QString unit,bool notify){
//    printf("%s [%s] = %s\r\n",__func__,
//           value.toStdString().c_str(),
//           unit.toStdString().c_str());
    if(_paramsMap.keys().contains(name)){
        int index = _paramsMap[name];
        _paramsModel[index]->setValue(value);
        _paramsModel[index]->setUnit(unit);
    }else{
        _paramsMap[name] = _paramsModel.length();
        Fact* fact = new Fact(nullptr);
        fact->setValue(value);
        fact->setName(name);
        fact->setUnit(unit);
        _paramsModel.append(fact);
    }
//    if(notify){
//        Q_EMIT paramsModelChanged();
//    }
}
void Vehicle::_sendMessageOnLink(IOFlightController *link, mavlink_message_t message)
{
//    printf("%s message.msgid[%d] message.compid[%d]\r\n", __func__, message.msgid, message.compid);
    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
    int len = mavlink_msg_to_send_buffer(buffer, &message);
//#ifdef DEBUG

//    printf("%s [compid=%d msgid=%d]\r\n", __func__, message.compid, message.msgid);
//#endif
    link->m_linkInterface->writeBytesSafe((const char *)buffer, len);
    _messagesSent++;

    Q_EMIT messagesSentChanged();
    // log packet send
    uint8_t buf[MAVLINK_MAX_PACKET_LEN+sizeof(quint64)];
    // Write the uint64 time in microseconds in big endian format before the message.
    // This timestamp is saved in UTC time. We are only saving in ms precision because
    // getting more than this isn't possible with Qt without a ton of extra code.
    quint64 time = static_cast<quint64>(QDateTime::currentMSecsSinceEpoch() * 1000);
    qToBigEndian(time, buf);
    // Then write the message to the buffer
    int length = mavlink_msg_to_send_buffer(buf + sizeof(quint64), &message);

    // Determine how many bytes were written by adding the timestamp size to the message size
    length += sizeof(quint64);
    QByteArray b(reinterpret_cast<const char*>(buf), len);
    LogController::writeBinaryLog(_logFile,b);
}

void Vehicle::_mavlinkMessageReceived(mavlink_message_t message)
{
#ifdef DEBUG_MESSAGE_RECV
    printf("Vehicle::_mavlinkMessageReceived message.seq[%d] - id[%d]\r\n", message.seq, message.msgid);
#endif
    //-- Check link status
    _messagesReceived++;
    messagesReceivedChanged();

    if (!_heardFrom) {
        if (message.msgid == MAVLINK_MSG_ID_HEARTBEAT) {
            _heardFrom = true;
            _compID = message.compid;
            _messageSeq = message.seq + 1;
        }
    } else {
        if (_compID == message.compid) {
            uint16_t seq_received = (uint16_t)message.seq;
            uint16_t packet_lost_count = 0;

            //-- Account for overflow during packet loss
            if (seq_received < _messageSeq) {
                packet_lost_count = (seq_received + 255) - _messageSeq;
            } else {
                packet_lost_count = seq_received - _messageSeq;
            }

            _messageSeq = message.seq + 1;
            _messagesLost += packet_lost_count;

            if (packet_lost_count)
                messagesLostChanged();
        }
    }

    switch (message.msgid) {
    case MAVLINK_MSG_ID_TIMESYNC:
        break;

    case MAVLINK_MSG_ID_HOME_POSITION:
        _handleHomePosition(message);
        break;

    case MAVLINK_MSG_ID_HEARTBEAT:
        _handleHeartbeat(message);
        break;

    case MAVLINK_MSG_ID_RADIO_STATUS:
        _handleRadioStatus(message);
        break;

    case MAVLINK_MSG_ID_RC_CHANNELS:
        _handleRCChannels(message);
        _handleRCIn(message);
        break;
    case MAVLINK_MSG_ID_RC_CHANNELS_RAW:
        _handleRCChannelsRaw(message);
        break;
    case MAVLINK_MSG_ID_SERVO_OUTPUT_RAW:
        _handleServoOut(message);
        break;
    case MAVLINK_MSG_ID_BATTERY_STATUS:
        _handleBatteryStatus(message);
        break;

    case MAVLINK_MSG_ID_BATTERY2:
        _handleBattery2Status(message);
        break;

    case MAVLINK_MSG_ID_SYS_STATUS:
        _handleSysStatus(message);
        break;

    case MAVLINK_MSG_ID_RAW_IMU:
        mavlinkRawImu(message);
        break;

    case MAVLINK_MSG_ID_SCALED_IMU:
        mavlinkScaledImu1(message);
        break;

    case MAVLINK_MSG_ID_SCALED_IMU2:
        mavlinkScaledImu2(message);
        break;

    case MAVLINK_MSG_ID_SCALED_IMU3:
        mavlinkScaledImu3(message);
        break;

    case MAVLINK_MSG_ID_VIBRATION:
        _handleVibration(message);
        break;
    case MAVLINK_MSG_ID_PMU:
        _handlePMU(message);
        break;
    case MAVLINK_MSG_ID_PW:
        _handlePW(message);
        break;
    case MAVLINK_MSG_ID_ECU:
        _handleECU(message);
        break;
    case MAVLINK_MSG_ID_AUX_ADC:
        _handleAUX_ADC(message);
        break;

    case MAVLINK_MSG_ID_COMMAND_ACK:
        _handleCommandAck(message);
        break;

    //    case MAVLINK_MSG_ID_COMMAND_LONG:
    //        _handleCommandLong(message);
    //        break;
    case MAVLINK_MSG_ID_AUTOPILOT_VERSION:
        _handleAutopilotVersion(message);
        break;

    case MAVLINK_MSG_ID_PROTOCOL_VERSION:
        _handleProtocolVersion(message);
        break;
    case MAVLINK_MSG_ID_WIND_COV:
        _handleWindCov(message);
        break;
    case MAVLINK_MSG_ID_RANGEFINDER:
        _handleRangeFinder(message);
        break;
    //    case MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS:
    //        _handleHilActuatorControls(message);
    //        break;
    //    case MAVLINK_MSG_ID_LOGGING_DATA:
    //        _handleMavlinkLoggingData(message);
    //        break;
    //    case MAVLINK_MSG_ID_LOGGING_DATA_ACKED:
    //        _handleMavlinkLoggingDataAcked(message);
    //        break;
    case MAVLINK_MSG_ID_GPS_RAW_INT:
        _handleGpsRawInt(message);
        break;

    case MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
        _handleGlobalPositionInt(message);
        break;

    //    case MAVLINK_MSG_ID_ALTITUDE:
    //        _handleAltitude(message);
    //        break;
    case MAVLINK_MSG_ID_VFR_HUD:
        _handleVfrHud(message);
        break;

    case MAVLINK_MSG_ID_SCALED_PRESSURE:
        _handleScaledPressure(message);
        break;
    case MAVLINK_MSG_ID_SCALED_PRESSURE2:
        _handleScaledPressure2(message);
        break;
    case MAVLINK_MSG_ID_SCALED_PRESSURE3:
        _handleScaledPressure3(message);
        break;
    //    case MAVLINK_MSG_ID_CAMERA_IMAGE_CAPTURED:
    //        _handleCameraImageCaptured(message);
    //        break;
    //    case MAVLINK_MSG_ID_ADSB_VEHICLE:
    //        _handleADSBVehicle(message);
    //        break;
    case MAVLINK_MSG_ID_HIGH_LATENCY2:
        _handleHighLatency2(message);
        break;

    case MAVLINK_MSG_ID_ATTITUDE:
        _handleAttitude(message);
        break;

    //    case MAVLINK_MSG_ID_ATTITUDE_QUATERNION:
    //        _handleAttitudeQuaternion(message);
    //        break;
    case MAVLINK_MSG_ID_ATTITUDE_TARGET:
        _handleAttitudeTarget(message);
        break;
//    case MAVLINK_MSG_ID_DISTANCE_SENSOR:
//        _handleDistanceSensor(message);
//        break;
//    case MAVLINK_MSG_ID_ESTIMATOR_STATUS:
//        _handleEstimatorStatus(message);
//        break;
    case MAVLINK_MSG_ID_STATUSTEXT:
        _handleStatusText(message, false /* longVersion */);
        break;
    case MAVLINK_MSG_ID_STATUSTEXT_LONG:
        _handleStatusText(message, true /* longVersion */);
//        break;
//    case MAVLINK_MSG_ID_ORBIT_EXECUTION_STATUS:
//        _handleOrbitExecutionStatus(message);
//        break;

//    case MAVLINK_MSG_ID_PING:
//        _handlePing(message);
//        break;
    case MAVLINK_MSG_ID_EKF_STATUS_REPORT:
    {
        _handleEKFState(message);
        break;
    }

    case MAVLINK_MSG_ID_RPM: {
        _handleRPMEngine(message);
        break;
    }

    case MAVLINK_MSG_ID_NAV_CONTROLLER_OUTPUT: {
        _handleNAVControllerOutput(message);
        break;
    }

    //    case MAVLINK_MSG_ID_SERIAL_CONTROL:
    //    {
    //        mavlink_serial_control_t ser;
    //        mavlink_msg_serial_control_decode(&message, &ser);
    //    }
    //        break;
    case MAVLINK_MSG_ID_WIND:
        _handleWind(message);
        break;
    }

    Q_EMIT mavlinkMessageReceived(message);
}
void Vehicle::_sendGCSHeartbeat(void)
{
    mavlink_message_t msg;
    uint16_t len;
    mavlink_msg_heartbeat_pack_chan(
        m_com->systemId(),
        m_com->componentId(),
        m_com->mavlinkChannel(),
        &msg,
        MAV_TYPE_GCS,
        MAV_AUTOPILOT_INVALID,
        MAV_MODE_MANUAL_ARMED,
        0,
        MAV_STATE_ACTIVE);
    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
    len = mavlink_msg_to_send_buffer(buffer, &msg);
    m_com->m_linkInterface->writeBytesSafe((const char *)(buffer), len);
    _countHeartBeat ++;
    if(_countHeartBeat >=8 ){
        if (m_firmwarePlugin != nullptr)
            m_firmwarePlugin->initializeVehicle(this);
        _countHeartBeat = 0;
    }
}
void Vehicle::_checkCameraLink(void){
//    printf("_cameraLink last vs recv [%d - %d]\r\n",_cameraLinkLast,_linkHeartbeatRecv);
    if(_linkHeartbeatRecv - _cameraLinkLast < 1){
        setLink(false);
        _cameraLinkLast = 0;
        _linkHeartbeatRecv = 0;
        if(m_uav == nullptr && m_com != nullptr){
//            printf("Reconnect link to UAV\r\n");
            m_com->disConnectLink();
            m_com->connectLink();
        }
    }else{

        setLink(true);
    }
    _cameraLinkLast = _linkHeartbeatRecv;
}
void Vehicle::requestDataStream(int messageID, int hz, int enable)
{
    mavlink_message_t msg;
    uint16_t len;
    mavlink_request_data_stream_t   dataStream;
    memset(&dataStream, 0, sizeof(dataStream));
    dataStream.req_stream_id = messageID;
    dataStream.req_message_rate = hz;
    dataStream.start_stop = enable;  // start
    dataStream.target_system = id();
    dataStream.target_component = _defaultComponentId;
    mavlink_msg_request_data_stream_encode_chan(m_com->systemId(),
            m_com->componentId(),
            m_com->mavlinkChannel(),
            &msg,
            &dataStream);
    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
    len = mavlink_msg_to_send_buffer(buffer, &msg);
    m_com->m_linkInterface->writeBytesSafe((const char *)(buffer), len);
}
void Vehicle::_startPlanRequest(void)
{
    printf("%s m_paramsController->missingParameters() = %s\r\n",__func__,
           m_paramsController->missingParameters()?"true":"false");
    if(!_requestPlanAfterParams && !m_paramsController->missingParameters())
        m_planController->loadFromVehicle();
    if(!_requestPlanAfterParams)
        _requestPlanAfterParams = true;
}
void Vehicle::_sendQGCTimeToVehicle(void)
{
    mavlink_message_t       msg;
    mavlink_system_time_t   cmd;
    // Timestamp of the master clock in microseconds since UNIX epoch.
    cmd.time_unix_usec = QDateTime::currentDateTime().currentMSecsSinceEpoch() * 1000;
    // Timestamp of the component clock since boot time in milliseconds (Not necessary).
    cmd.time_boot_ms = 0;
    mavlink_msg_system_time_encode_chan(m_com->systemId(),
                                        m_com->componentId(),
                                        m_com->mavlinkChannel(),
                                        &msg,
                                        &cmd);
    sendMessageOnLink(m_com, msg);
}
void Vehicle::_sendGetParams(void)
{
    //    printf("_sendGetParams\r\n");
    mavlink_message_t msg;
    uint16_t len;
    mavlink_msg_param_request_list_pack_chan(m_com->systemId(),
            m_com->componentId(),
            m_com->mavlinkChannel(),
            &msg,
            this->id(),
            MAV_COMP_ID_ALL);
    //    mavlink_msg_request_data_stream_pack_chan(m_com->systemId(),
    //                                             m_com->componentId(),
    //                                             m_com->mavlinkChannel(),
    //                                             &msg,
    //                                             this->id(),
    //                                             m_com->componentId(),
    //                                             MAVLINK_MSG_ID_EKF_STATUS_REPORT,
    //                                             3,1);
    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
    len = mavlink_msg_to_send_buffer(buffer, &msg);
    m_com->m_linkInterface->writeBytesSafe((const char *)(buffer), len);
}
void Vehicle::_updateArmed(bool armed)
{
    if (_armed != armed) {
        _armed = armed;
    }

    Q_EMIT armedChanged(_armed);
}
void Vehicle::_handleHomePosition(mavlink_message_t &message)
{

    mavlink_home_position_t homePos;
    mavlink_msg_home_position_decode(&message, &homePos);
    _homePosition.setLatitude(homePos.latitude / 10000000.0);
    _homePosition.setLongitude(homePos.longitude / 10000000.0);
//    QGeoCoordinate newHomePosition(homePos.latitude / 10000000.0,
//                                   homePos.longitude / 10000000.0,
//                                   homePos.altitude / 1000.0);
//    _setHomePosition(newHomePosition);
    Q_EMIT homePositionChanged(_homePosition);
//#ifdef DEBUG_FUNC
    printf("%s altitude = %f\r\n", __func__,static_cast<double>(homePos.altitude / 1000.0));
//#endif
}
void Vehicle::_handleHeartbeat(mavlink_message_t &message)
{
    mavlink_heartbeat_t heartbeat;
    mavlink_msg_heartbeat_decode(&message, &heartbeat);
//        printf("=============================\r\n");
//        printf("heartbeat.base_mode=%d\r\n",heartbeat.base_mode);
//        printf("heartbeat.custom_mode=%d\r\n",heartbeat.custom_mode);
//        printf("heartbeat.type=%d\r\n",heartbeat.type);
//        printf("heartbeat.system_status=%d\r\n",heartbeat.system_status);
//        printf("heartbeat.autopilot=%d\r\n",heartbeat.autopilot);
//        printf("heartbeat.mavlink_version=%d\r\n",heartbeat.mavlink_version);

    if (heartbeat.type != MAV_TYPE_GCS && heartbeat.type != MAV_AUTOPILOT_INVALID
        && (heartbeat.base_mode != _base_mode || heartbeat.custom_mode != _custom_mode)
            ) {
        bool newArmed = heartbeat.base_mode & MAV_MODE_FLAG_DECODE_POSITION_SAFETY;
        _updateArmed(newArmed);
        _base_mode = heartbeat.base_mode;
        _custom_mode = heartbeat.custom_mode;
        flightModeChanged(flightMode());
    }

    if (heartbeat.type != MAV_TYPE_GCS && heartbeat.type != MAV_AUTOPILOT_INVALID) {
        //        printf("_handleHeartbeat\r\n");

//        if(_landed != ((MAV_STATE)(heartbeat.system_status) == MAV_STATE::MAV_STATE_STANDBY))
        {
            _landed = ((MAV_STATE)(heartbeat.system_status) == MAV_STATE::MAV_STATE_STANDBY);
            _setPropertyValue("Landed",((MAV_STATE)(heartbeat.system_status) == MAV_STATE::MAV_STATE_STANDBY)?"True":"False","");
            Q_EMIT landedChanged();
        }

        if (_firmwareType != static_cast<MAV_AUTOPILOT>(heartbeat.autopilot)) {
            _firmwareType = static_cast<MAV_AUTOPILOT>(heartbeat.autopilot);
        }
        _linkHeartbeatRecv ++;
        if (_vehicleType != static_cast<VEHICLE_MAV_TYPE>(heartbeat.type)) {
            setVehicleType(static_cast<VEHICLE_MAV_TYPE>(heartbeat.type));
            printf("Change firmware plugin to _vehicleType=%d\r\n", _vehicleType);
            FirmwarePlugin *newFimware =
                        m_firmwarePluginManager->firmwarePluginForAutopilot(
                        this,_firmwareType, static_cast<MAV_TYPE>(heartbeat.type));
            if (newFimware != nullptr) {
                if (m_firmwarePlugin != nullptr)
                    delete m_firmwarePlugin;

                m_firmwarePlugin = newFimware;
                m_firmwarePlugin->initializeVehicle(this);
                m_paramsController->refreshAllParameters();
                Q_EMIT flightModesChanged();
                Q_EMIT flightModesOnAirChanged();
                flightModeChanged(flightMode());
            }
        }
    }
}
void Vehicle::_handleRadioStatus(mavlink_message_t &message)
{
    //-- Process telemetry status message
    mavlink_radio_status_t rstatus;
    mavlink_msg_radio_status_decode(&message, &rstatus);
    int rssi    = rstatus.rssi;
    int remrssi = rstatus.remrssi;
    int lnoise = (int)(int8_t)rstatus.noise;
    int rnoise = (int)(int8_t)rstatus.remnoise;

    //-- 3DR Si1k radio needs rssi fields to be converted to dBm
    if (message.sysid == '3' && message.compid == 'D') {
        /* Per the Si1K datasheet figure 23.25 and SI AN474 code
         * samples the relationship between the RSSI register
         * and received power is as follows:
         *
         *                       10
         * inputPower = rssi * ------ 127
         *                       19
         *
         * Additionally limit to the only realistic range [-120,0] dBm
         */
        rssi    = qMin(qMax(qRound(static_cast<qreal>(rssi)    / 1.9 - 127.0), - 120), 0);
        remrssi = qMin(qMax(qRound(static_cast<qreal>(remrssi) / 1.9 - 127.0), - 120), 0);
    } else {
        rssi    = (int)(int8_t)rstatus.rssi;
        remrssi = (int)(int8_t)rstatus.remrssi;
    }

    //-- Check for changes
    if (_telemetryLRSSI != rssi) {
        _telemetryLRSSI = rssi;
    }
    Q_EMIT rssiChanged();
    if(m_uav!=nullptr){
        m_uav->_setPropertyValue("PTU_RSSI",QString::fromStdString(std::to_string(rssi)),"dBm");
    }
    if (_telemetryRRSSI != remrssi) {
        _telemetryRRSSI = remrssi;
    }

    if (_telemetryRXErrors != rstatus.rxerrors) {
        _telemetryRXErrors = rstatus.rxerrors;
    }

    if (_telemetryFixed != rstatus.fixed) {
        _telemetryFixed = rstatus.fixed;
    }

    if (_telemetryTXBuffer != rstatus.txbuf) {
        _telemetryTXBuffer = rstatus.txbuf;
    }

    if (_telemetryLNoise != lnoise) {
        _telemetryLNoise = lnoise;
    }

    if (_telemetryRNoise != rnoise) {
        _telemetryRNoise = rnoise;
    }
}
void Vehicle::_handleScaledPressure(mavlink_message_t& message) {
    mavlink_scaled_pressure_t pressure;
    mavlink_msg_scaled_pressure_decode(&message, &pressure);
    _temperature = pressure.temperature;
    _pressABS = pressure.press_abs;
    Q_EMIT temperatureChanged();
    Q_EMIT pressABSChanged();
    if(m_uav!= nullptr){
        m_uav->_setPropertyValue("PTU_Temperature",QString::fromStdString(std::to_string(_temperature)),"cdegC");
        m_uav->_setPropertyValue("PTU_Press",QString::fromStdString(std::to_string(_pressABS)),"hPa");
    }
#ifdef DEBUG_FUNC
   printf("%s temp[%d] pressABS[%f]\r\n",__func__,_temperature,_pressABS);
#endif
}

void Vehicle::_handleScaledPressure2(mavlink_message_t& message) {
    mavlink_scaled_pressure2_t pressure;
    mavlink_msg_scaled_pressure2_decode(&message, &pressure);
}

void Vehicle::_handleScaledPressure3(mavlink_message_t& message) {
    mavlink_scaled_pressure3_t pressure;
    mavlink_msg_scaled_pressure3_decode(&message, &pressure);
}
void Vehicle::_handleRCChannels(mavlink_message_t& message)
{
    mavlink_rc_channels_t channels;
    mavlink_msg_rc_channels_decode(&message, &channels);
}
void Vehicle::_handleRCIn(mavlink_message_t& message)
{
    mavlink_rc_channels_t rcIn;
    mavlink_msg_rc_channels_decode(&message, &rcIn);
    _setPropertyValue("RCIN_chan1",QString::fromStdString(std::to_string(rcIn.chan1_raw)),"us");
    _setPropertyValue("RCIN_chan2",QString::fromStdString(std::to_string(rcIn.chan2_raw)),"us");
    _setPropertyValue("RCIN_chan3",QString::fromStdString(std::to_string(rcIn.chan3_raw)),"us");
    _setPropertyValue("RCIN_chan4",QString::fromStdString(std::to_string(rcIn.chan4_raw)),"us");
    _setPropertyValue("RCIN_chan5",QString::fromStdString(std::to_string(rcIn.chan5_raw)),"us");
    _setPropertyValue("RCIN_chan6",QString::fromStdString(std::to_string(rcIn.chan6_raw)),"us");
    _setPropertyValue("RCIN_chan7",QString::fromStdString(std::to_string(rcIn.chan7_raw)),"us");
    _setPropertyValue("RCIN_chan8",QString::fromStdString(std::to_string(rcIn.chan8_raw)),"us");
    _setPropertyValue("RCIN_chan9",QString::fromStdString(std::to_string(rcIn.chan9_raw)),"us");
    _setPropertyValue("RCIN_chan10",QString::fromStdString(std::to_string(rcIn.chan10_raw)),"us");
    _setPropertyValue("RCIN_chan11",QString::fromStdString(std::to_string(rcIn.chan11_raw)),"us");
    _setPropertyValue("RCIN_chan12",QString::fromStdString(std::to_string(rcIn.chan12_raw)),"us");
    _setPropertyValue("RCIN_chan13",QString::fromStdString(std::to_string(rcIn.chan13_raw)),"us");
    _setPropertyValue("RCIN_chan14",QString::fromStdString(std::to_string(rcIn.chan14_raw)),"us");
    _setPropertyValue("RCIN_chan15",QString::fromStdString(std::to_string(rcIn.chan15_raw)),"us");
    _setPropertyValue("RCIN_chan16",QString::fromStdString(std::to_string(rcIn.chan16_raw)),"us");
    _setPropertyValue("RCIN_chan17",QString::fromStdString(std::to_string(rcIn.chan17_raw)),"us");
    _setPropertyValue("RCIN_chan18",QString::fromStdString(std::to_string(rcIn.chan18_raw)),"us");
}

void Vehicle::_handleServoOut(mavlink_message_t& message)
{
    mavlink_servo_output_raw_t servoOut;
    mavlink_msg_servo_output_raw_decode(&message, &servoOut);
    _setPropertyValue("ServoOut_1",QString::fromStdString(std::to_string(servoOut.servo1_raw)),"us");
    _setPropertyValue("ServoOut_2",QString::fromStdString(std::to_string(servoOut.servo2_raw)),"us");
    _setPropertyValue("ServoOut_3",QString::fromStdString(std::to_string(servoOut.servo3_raw)),"us");
    _setPropertyValue("ServoOut_4",QString::fromStdString(std::to_string(servoOut.servo4_raw)),"us");
    _setPropertyValue("ServoOut_5",QString::fromStdString(std::to_string(servoOut.servo5_raw)),"us");
    _setPropertyValue("ServoOut_6",QString::fromStdString(std::to_string(servoOut.servo6_raw)),"us");
    _setPropertyValue("ServoOut_7",QString::fromStdString(std::to_string(servoOut.servo7_raw)),"us");
    _setPropertyValue("ServoOut_8",QString::fromStdString(std::to_string(servoOut.servo8_raw)),"us");
    _setPropertyValue("ServoOut_9",QString::fromStdString(std::to_string(servoOut.servo9_raw)),"us");
    _setPropertyValue("ServoOut_10",QString::fromStdString(std::to_string(servoOut.servo10_raw)),"us");
    _setPropertyValue("ServoOut_11",QString::fromStdString(std::to_string(servoOut.servo11_raw)),"us");
    _setPropertyValue("ServoOut_12",QString::fromStdString(std::to_string(servoOut.servo12_raw)),"us");
    _setPropertyValue("ServoOut_13",QString::fromStdString(std::to_string(servoOut.servo13_raw)),"us");
    _setPropertyValue("ServoOut_14",QString::fromStdString(std::to_string(servoOut.servo14_raw)),"us");
    _setPropertyValue("ServoOut_15",QString::fromStdString(std::to_string(servoOut.servo15_raw)),"us");
    _setPropertyValue("ServoOut_16",QString::fromStdString(std::to_string(servoOut.servo16_raw)),"us");
}
void Vehicle::_handleRCChannelsRaw(mavlink_message_t& message)
{
    // We handle both RC_CHANNLES and RC_CHANNELS_RAW since different firmware will only
    // send one or the other.

    mavlink_rc_channels_raw_t channels;
    mavlink_msg_rc_channels_raw_decode(&message, &channels);

}
void Vehicle::_handleBatteryStatus(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_battery_status_t bat_status;
    mavlink_msg_battery_status_decode(&message, &bat_status);
    int cellCount = 0;
    float voltage = 0;
    float wats = 0;

    for (int i = 0; i < 10; i++) {
//        printf("bat_status.voltages[%d] = 0x%04x\r\n",i,bat_status.voltages[i]);
        if (bat_status.voltages[i] != UINT16_MAX) {
//            printf("bat_status.voltages[%d] = %f\r\n",i,bat_status.voltages[i]/1000.0f);
            cellCount++;
            voltage += bat_status.voltages[i];
        }
    }

    if (cellCount == 0) {
        cellCount = -1;
    }

//    printf("batery[%d] [%d]cell - %.02fV - %.02fW - %.02fA\r\n",
//           bat_status.id,cellCount,
//           voltage/static_cast<float>(cellCount)/1000.f,
//           static_cast<float>(bat_status.current_consumed) / 1000.0f,
//           static_cast<float>(bat_status.current_battery)/100.f);
//    _batteryVoltage = voltage / static_cast<float>(cellCount) / 1000.f;
//    _batteryAmpe = static_cast<float>(bat_status.current_battery) / 100.f;
//    Q_EMIT batteryVoltageChanged();
//    Q_EMIT batteryAmpeChanged();
}
void Vehicle::_handleBattery2Status(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_battery2_t bat2_status;
    mavlink_msg_battery2_decode(&message, &bat2_status);
}
void Vehicle::_handleSysStatus(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_sys_status_t sysStatus;
    mavlink_msg_sys_status_decode(&message, &sysStatus);

    if (sysStatus.current_battery == -1) {

    } else {
        _batteryAmpe = static_cast<float>(sysStatus.current_battery) / 100.f;
        Q_EMIT batteryAmpeChanged();
    }
    if (sysStatus.voltage_battery == UINT16_MAX) {
        m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Voltage unavailable");
    } else {
        _batteryVoltage = sysStatus.voltage_battery / 1000.f;
        _setPropertyValue("Current",
                                 QString::fromStdString(std::to_string(sysStatus.voltage_battery)),
                                 "mA");
        Q_EMIT batteryVoltageChanged();
    }

    if (sysStatus.battery_remaining > 0) {
        if(sysStatus.battery_remaining < 20)
        m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,
            QString(tr("%1 low battery: %2 percent remaining")).arg(id()).arg(sysStatus.battery_remaining));
    }

    if (_onboardControlSensorsPresent != sysStatus.onboard_control_sensors_present) {
        _onboardControlSensorsPresent = sysStatus.onboard_control_sensors_present;
        printf("_onboardControlSensorsPresent=[0X%08X]\r\n",_onboardControlSensorsPresent);
    }
    if (_onboardControlSensorsEnabled != sysStatus.onboard_control_sensors_enabled) {
        _onboardControlSensorsEnabled = sysStatus.onboard_control_sensors_enabled;
        printf("_onboardControlSensorsEnabled=[0X%08X]\r\n",_onboardControlSensorsEnabled);
    }
    if (_onboardControlSensorsHealth != sysStatus.onboard_control_sensors_health) {
        _onboardControlSensorsHealth = sysStatus.onboard_control_sensors_health;
        printf("_onboardControlSensorsHealth=[0X%08X]\r\n",_onboardControlSensorsHealth);
    }

    // ArduPilot firmare has a strange case when ARMING_REQUIRE=0. This means the vehicle is always armed but the motors are not
    // really powered up until the safety button is pressed. Because of this we can't depend on the heartbeat to tell us the true
    // armed (and dangerous) state. We must instead rely on SYS_STATUS telling us that the motors are enabled.
//    if (apmFirmware() && _apmArmingNotRequired()) {
//        _updateArmed(_onboardControlSensorsEnabled & MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS);
//    }

    uint32_t newSensorsUnhealthy = _onboardControlSensorsEnabled & ~_onboardControlSensorsHealth;
    if (newSensorsUnhealthy != _onboardControlSensorsUnhealthy) {
        _onboardControlSensorsUnhealthy = newSensorsUnhealthy;
//        Q_EMIT unhealthySensorsChanged();
        if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensor3dMag){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. Magnetometer issues. Check console.");
        }else if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensor3dAccel){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. Accelerometer issues. Check console.");
        }else if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensor3dGyro){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. Gyroscope issues. Check console.");
        }else if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensorAbsolutePressure){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. Barometer issues. Check console.");
        }else if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensorDifferentialPressure){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. Airspeed issues. Check console.");
        }else if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensorAHRS){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. AHRS issues. Check console.");
        }else if(_onboardControlSensorsUnhealthy && Vehicle::SysStatusSensorGPS){
            m_uas->handleTextMessage(0,m_com->componentId(),MAV_SEVERITY_ERROR,"Failure. GPS issues. Check console.");
        }
    }
}
void Vehicle::_handleGpsRawInt(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_gps_raw_int_t gpsRawInt;
    mavlink_msg_gps_raw_int_decode(&message, &gpsRawInt);
    _gpsRawIntMessageAvailable = true;
//    QGeoCoordinate newPosition(gpsRawInt.lat  / (double)1E7, gpsRawInt.lon / (double)1E7, gpsRawInt.alt  / 1000.0);
//    printf("%s %f,%f,%f\r\n", __func__,
//           newPosition.latitude(),newPosition.longitude(),
//           static_cast<float>(gpsRawInt.alt_ellipsoid) / 1000.0);
//    if (newPosition != _coordinate) {
//        _coordinate = newPosition;
//        Q_EMIT coordinateChanged(_coordinate);
//        float _tmpHeadingToHome = static_cast<float>(coordinate().azimuthTo(homePosition()));

//        if (_tmpHeadingToHome - _headingToHome > 0 ||
//            _tmpHeadingToHome - _headingToHome < 0) {
//            _headingToHome = _tmpHeadingToHome;
//            Q_EMIT headingToHomeChanged();
//        }

//        float _tmpDistanceToHome = static_cast<float>(coordinate().distanceTo(homePosition()));

//        if (_tmpDistanceToHome - _distanceToHome > 0 ||
//            _tmpDistanceToHome - _distanceToHome < 0) {
//            _distanceToHome = _tmpDistanceToHome;
//            Q_EMIT distanceToHomeChanged();
//        }

//        _altitudeAMSL = static_cast<float>(gpsRawInt.alt) / 1000.0;
//        Q_EMIT altitudeAMSLChanged();
//    }

    if (gpsRawInt.fix_type != _gpsFixedType) {
        _gpsFixedType = gpsRawInt.fix_type;
        Q_EMIT gpsChanged(gpsRawInt.fix_type > 1);
    }
    _latGPS = static_cast<float>(gpsRawInt.lat * 1e-7);
    _lonGPS = static_cast<float>(gpsRawInt.lon * 1e-7);
    _countGPS = static_cast<int>(gpsRawInt.satellites_visible == 255 ? 0 : gpsRawInt.satellites_visible);
    _hdopGPS = static_cast<float>(gpsRawInt.eph == UINT16_MAX ? std::numeric_limits<double>::quiet_NaN() : gpsRawInt.eph / 100.0);
    _vdopGPS = static_cast<float>(gpsRawInt.epv == UINT16_MAX ? std::numeric_limits<double>::quiet_NaN() : gpsRawInt.epv / 100.0);
    _courseOverGroundGPS = static_cast<float>(gpsRawInt.cog == UINT16_MAX ? std::numeric_limits<double>::quiet_NaN() : gpsRawInt.cog / 100.0);
    QMap<int,QString> lockGPSMap;
    lockGPSMap[0] =  "No GPS connected";
    lockGPSMap[1] =  "No position information, GPS is connected";
    lockGPSMap[2] =  "2D position";
    lockGPSMap[3] =  "3D position";
    lockGPSMap[4] =  "DGPS/SBAS aided 3D position";
    lockGPSMap[5] =  "RTK float, 3D position";
    lockGPSMap[6] =  "RTK Fixed, 3D position";
    lockGPSMap[7] =  "Static fixed, typically used for base stations";
    lockGPSMap[8] =  "PPP, 3D position";
    if(lockGPSMap.contains(gpsRawInt.fix_type)){
        _lockGPS = lockGPSMap[gpsRawInt.fix_type];
        Q_EMIT lockGPSChanged();
    }
    Q_EMIT latGPSChanged();
    Q_EMIT lonGPSChanged();
    Q_EMIT hdopGPSChanged();
    Q_EMIT vdopGPSChanged();
    Q_EMIT courseOverGroundGPSChanged();
    Q_EMIT countGPSChanged();
    _setPropertyValue("GPSHdop",QString::fromStdString(std::to_string(_hdopGPS)),"");
    _setPropertyValue("GPSSatCount",QString::fromStdString(std::to_string(_countGPS)),"");
}
void Vehicle::_handleGlobalPositionInt(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_global_position_int_t globalPositionInt;
    mavlink_msg_global_position_int_decode(&message, &globalPositionInt);

    QGeoCoordinate newPosition(globalPositionInt.lat  / (double)1E7, globalPositionInt.lon / (double)1E7, globalPositionInt.alt  / 1000.0);
//    printf("%s %f,%f\r\n", __func__,newPosition.latitude(),newPosition.longitude());
    if (newPosition != _coordinate) {
        _coordinate = newPosition;
        if(m_uav!= nullptr){
            m_uav->_setPropertyValue("LatHome",
                                     QString::fromStdString(std::to_string(_coordinate.latitude())),
                                     "deg");
            m_uav->_setPropertyValue("LongHome",
                                     QString::fromStdString(std::to_string(_coordinate.longitude())),
                                     "deg");
        }else{
            _setPropertyValue("Latitude",
                                     QString::fromStdString(std::to_string(_coordinate.latitude())),
                                     "deg");
            _setPropertyValue("Longitude",
                                     QString::fromStdString(std::to_string(_coordinate.longitude())),
                                     "deg");
        }

       Q_EMIT coordinateChanged(_coordinate);
       float _tmpHeadingToHome = static_cast<float>(coordinate().azimuthTo(homePosition()));
       if(_tmpHeadingToHome - _headingToHome > 0 ||
               _tmpHeadingToHome - _headingToHome < 0){
           _headingToHome = _tmpHeadingToHome;
           Q_EMIT headingToHomeChanged();
       }
       float _tmpDistanceToHome = static_cast<float>(coordinate().distanceTo(homePosition()));
       if(_tmpDistanceToHome - _distanceToHome > 0 ||
               _tmpDistanceToHome - _distanceToHome < 0){
           _distanceToHome = _tmpDistanceToHome;
           Q_EMIT distanceToHomeChanged();
       }
       _altitudeAMSL = static_cast<float>(globalPositionInt.alt) / 1000.0;
       Q_EMIT altitudeAMSLChanged();
    }
    if (!_globalPositionIntMessageAvailable) {
        _globalPositionIntMessageAvailable = true;
    }

    _altitudeRelative = static_cast<float>(globalPositionInt.relative_alt) / 1000.0f;
    _setPropertyValue("AltitudeAGL",QString::fromStdString(std::to_string(_altitudeRelative)),"m");
    Q_EMIT altitudeRelativeChanged();
    if (!_gpsRawIntMessageAvailable) {
        _gpsRawIntMessageAvailable = true;
    }

    _altitudeAMSL = static_cast<float>(globalPositionInt.alt) / 1000.0f;
    _setPropertyValue("AltitudeAMSL",QString::fromStdString(std::to_string(_altitudeAMSL)),"m");
    Q_EMIT altitudeAMSLChanged();
    if(m_uav!=nullptr){
        m_uav->_setPropertyValue("PTU_Alt",QString::fromStdString(std::to_string(globalPositionInt.relative_alt)),"m");
    }
#ifdef DEBUG_FUNC
    printf("%s altitude:%f\r\n", __func__, static_cast<float>(globalPositionInt.relative_alt) / 1000.0f);
#endif
}
void Vehicle::_handleAltitude(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_altitude_t altitude;
    mavlink_msg_altitude_decode(&message, &altitude);
//    printf("altitude.altitude_relative = %f\r\n",altitude.altitude_relative);
    // If data from GPS is available it takes precedence over ALTITUDE message
    if (!_globalPositionIntMessageAvailable) {
        _altitudeRelative = (altitude.altitude_relative);
        Q_EMIT altitudeRelativeChanged();

        if (!_gpsRawIntMessageAvailable) {
            _altitudeAMSL = (altitude.altitude_amsl);
            Q_EMIT altitudeAMSLChanged();
        }
    }
}
void Vehicle::_handleStatusText(mavlink_message_t& message, bool longVersion)
{

    QByteArray  b;
    QString     messageText;
    int         severity;

    if (longVersion) {
        b.resize(MAVLINK_MSG_STATUSTEXT_LONG_FIELD_TEXT_LEN+1);
        mavlink_msg_statustext_long_get_text(&message, b.data());
        severity = mavlink_msg_statustext_long_get_severity(&message);
    } else {
        b.resize(MAVLINK_MSG_STATUSTEXT_FIELD_TEXT_LEN+1);
        mavlink_msg_statustext_get_text(&message, b.data());
        severity = mavlink_msg_statustext_get_severity(&message);
    }
    b[b.length()-1] = '\0';
    messageText = QString(b);

    bool skipSpoken = false;
    bool ardupilotPrearm = messageText.startsWith(QStringLiteral("PreArm"));

    // If the message is NOTIFY or higher severity, or starts with a '#',
    // then read it aloud.
    if (messageText.startsWith("#") || severity <= MAV_SEVERITY_NOTICE) {
        messageText.remove("#");
        if (!skipSpoken) {

        }
    }
#ifdef DEBUG_FUNC
    printf("%s %s version %s\r\n",__func__,longVersion?"long":"short",messageText.toStdString().c_str());
#endif
    m_uas->handleTextMessage(0,m_com->componentId(),severity,messageText);
    if(ardupilotPrearm){
        setMessageSecurity("MSG_ERROR");
    }else{
        setMessageSecurity("MSG_INFO");
    }

    Q_EMIT uasChanged();

//    Q_EMIT textMessageReceived(id(), message.compid, severity, messageText);
}
void Vehicle::_handleVfrHud(mavlink_message_t& message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_vfr_hud_t vfrHud;
    mavlink_msg_vfr_hud_decode(&message, &vfrHud);
    _airSpeed = (qIsNaN(vfrHud.airspeed) ? 0 : vfrHud.airspeed);
    _groundSpeed = (qIsNaN(vfrHud.groundspeed) ? 0 : vfrHud.groundspeed);
    _climbSpeed = (qIsNaN(vfrHud.climb) ? 0 : vfrHud.climb);
    Q_EMIT airSpeedChanged();
    Q_EMIT groundSpeedChanged();
    Q_EMIT climbSpeedChanged();
    _setPropertyValue("AirSpeed",QString::fromStdString(std::to_string(_airSpeed*3.6)),"km/h");
    _setPropertyValue("GroundSpeed",QString::fromStdString(std::to_string(_groundSpeed*3.6)),"km/h");
    _setPropertyValue("ClimbSpeed",QString::fromStdString(std::to_string(_climbSpeed*3.6)),"km/h");
}
void Vehicle::_handleHighLatency2(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_high_latency2_t highLatency2;
    mavlink_msg_high_latency2_decode(&message, &highLatency2);
    //    QString previousFlightMode;
    //    if (_base_mode != 0 || _custom_mode != 0){
    //        // Vehicle is initialized with _base_mode=0 and _custom_mode=0. Don't pass this to flightMode() since it will complain about
    //        // bad modes while unit testing.
    //        previousFlightMode = flightMode();
    //    }
}
void Vehicle::_handleAttitude(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif

    if (_receivingAttitudeQuaternion) {
        return;
    }

    mavlink_attitude_t attitude;
    mavlink_msg_attitude_decode(&message, &attitude);
    _roll = (qRadiansToDegrees(attitude.roll));
    _pitch = (qRadiansToDegrees(attitude.pitch));
    _heading = (qRadiansToDegrees(attitude.yaw));
//    printf("Time: %s\r\n",FileController::get_time_stamp().c_str());
//    printf("_roll = %f\r\n",_roll);
//    printf("_pitch = %f\r\n",_pitch);
//    printf("_heading = %f\r\n",_heading);
    if (_heading < 0) {
        _heading = 360 + _heading;
    }

    Q_EMIT rollChanged();
    Q_EMIT pitchChanged();
    Q_EMIT headingChanged();
    if(m_uav!=nullptr){
        m_uav->_setPropertyValue("PTU_Heading",QString::fromStdString(std::to_string(_heading)),"deg");
    }
    _setPropertyValue("Roll",QString::fromStdString(std::to_string(_roll)),"deg");
    _setPropertyValue("Pitch",QString::fromStdString(std::to_string(_pitch)),"deg");
    _setPropertyValue("Yaw",QString::fromStdString(std::to_string(_heading)),"deg");
    //    _handleAttitudeWorker(attitude.roll, attitude.pitch, attitude.yaw);
}
void Vehicle::_handleAttitudeTarget(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_attitude_target_t attitudeTarget;
    mavlink_msg_attitude_target_decode(&message, &attitudeTarget);
    float roll, pitch, yaw;
    mavlink_quaternion_to_euler(attitudeTarget.q, &roll, &pitch, &yaw);
    _roll = (qRadiansToDegrees(roll));
    _pitch = (qRadiansToDegrees(pitch));
    _heading = (qRadiansToDegrees(yaw));

    if (_heading < 0) {
        _heading = 360 + _heading;
    }

    Q_EMIT rollChanged();
    Q_EMIT pitchChanged();
    Q_EMIT headingChanged();

}
void Vehicle::_handleEKFState(mavlink_message_t &message)
{
    mavlink_ekf_status_report_t ekf_status;
    mavlink_msg_ekf_status_report_decode(&message, &ekf_status);
#ifdef DEBUG_FUNC
    printf("%s compass_variance[%f] velocity_variance[%f] pos_horiz_variance[%f] pos_vert_variance[%f]\r\n",
           __func__,
           ekf_status.compass_variance,
           ekf_status.velocity_variance,
           ekf_status.pos_horiz_variance,
           ekf_status.pos_vert_variance);
#endif
    _setPropertyValue("EkfCompass",QString::fromStdString(std::to_string(ekf_status.compass_variance)),"");
    _setPropertyValue("EkfPosH",QString::fromStdString(std::to_string(ekf_status.velocity_variance)),"");
    _setPropertyValue("EkfPosV",QString::fromStdString(std::to_string(ekf_status.pos_horiz_variance)),"");
    _setPropertyValue("EkfVel",QString::fromStdString(std::to_string(ekf_status.pos_vert_variance)),"");
    float ekfs[4] = {ekf_status.compass_variance,
                     ekf_status.velocity_variance,
                     ekf_status.pos_horiz_variance,
                     ekf_status.pos_vert_variance
                    };

    for (int i = 0; i < sizeof(ekfs); i++) {
        if (ekfs[i] >= 0.8) {
            setEkfSignal("red");
            break;
        } else if (ekfs[i] >= 0.5) {
            setEkfSignal("orange");
            break;
        } else {
            setEkfSignal("green");
            break;
        }
    }
}
void Vehicle::_handleRPMEngine(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    mavlink_rpm_t rpm_status;
    mavlink_msg_rpm_decode(&message, &rpm_status);
    _engineSensor_1 = (rpm_status.rpm1);
    _engineSensor_2 = (rpm_status.rpm2);
    Q_EMIT engineSensor_1Changed();
    Q_EMIT engineSensor_2Changed();
    _setPropertyValue("RPM1",QString::fromStdString(std::to_string(rpm_status.rpm1)),"rpm");
    _setPropertyValue("RPM2",QString::fromStdString(std::to_string(rpm_status.rpm2)),"rpm");
}
void Vehicle::_handleNAVControllerOutput(mavlink_message_t &message)
{
    mavlink_nav_controller_output_t navControllerOutput;
    mavlink_msg_nav_controller_output_decode(&message, &navControllerOutput);
    setDistanceToCurrentWaypoint(navControllerOutput.wp_dist);
}
void Vehicle::_handleVibration(mavlink_message_t &message)
{
    mavlink_vibration_t vibration;
    mavlink_msg_vibration_decode(&message, &vibration);
#ifdef DEBUG_FUNC
    printf("%s [%f,%f,%f]\r\n", __func__,
           vibration.vibration_x,
           vibration.vibration_y,
           vibration.vibration_z);
#endif
    _setPropertyValue("VibeX",QString::fromStdString(std::to_string(vibration.vibration_x)),"m/ss");
    _setPropertyValue("VibeY",QString::fromStdString(std::to_string(vibration.vibration_y)),"m/ss");
    _setPropertyValue("VibeZ",QString::fromStdString(std::to_string(vibration.vibration_z)),"m/ss");
    if (vibration.vibration_x >= 60 || vibration.vibration_y >= 60 || vibration.vibration_z >= 60) {
        setVibeSignal("red");
    } else if (vibration.vibration_x >= 30 || vibration.vibration_y >= 30 || vibration.vibration_z >= 30) {
        setVibeSignal("orange");
    } else {
        setVibeSignal("green");
    }
}
void Vehicle::_handlePMU(mavlink_message_t &message)
{
    mavlink_pmu_t pmu;
    mavlink_msg_pmu_decode(&message, &pmu);
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    _setPropertyValue("I_BattA",
                   QString::fromStdString(std::to_string(pmu.IbattA)),"A");
    _setPropertyValue("I_BattB",
                   QString::fromStdString(std::to_string(pmu.IbattB)),"A");
    _setPropertyValue("V_BattA",
            QString::fromStdString(std::to_string(pmu.VbattA)),"V");
    _setPropertyValue("V_BattB",
            QString::fromStdString(std::to_string(pmu.VbattB)),"V");
    _setPropertyValue("PMU_Rpm",
            QString::fromStdString(std::to_string(pmu.PMU_RPM)),"rpm");
    _setPropertyValue("V_Batt12S",
            QString::fromStdString(std::to_string(pmu.Vbatt12S)),"V");
    _setPropertyValue("PMU_Temp",
            QString::fromStdString(std::to_string(pmu.env_temp)),"deg");
    _setPropertyValue("PMU_RH",
            QString::fromStdString(std::to_string(pmu.env_RH)),"");
    _setPropertyValue("PMU_Fuel_level",
            QString::fromStdString(std::to_string(pmu.Fuel_level)),"%");
    _setPropertyValue("PMU_Raw_fuel_level",
            QString::fromStdString(std::to_string(pmu.Raw_fuel_level)),"%");
    _setPropertyValue("PMU_data_status",
            QString::fromStdString(std::to_string(pmu.PMU_data_status)),"");
    _setPropertyValue("PMU_frame_ok",
            QString::fromStdString(std::to_string(pmu.PMU_frame_ok)),"");
    _setPropertyValue("PMU_com",
                      QString::fromStdString(std::to_string(pmu.PMU_com)),"");

}

void Vehicle::_handlePW(mavlink_message_t &message)
{
    mavlink_pw_t pw;
    mavlink_msg_pw_decode(&message, &pw);
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    _setPropertyValue("V_BattA",
                   QString::fromStdString(std::to_string(pw.VbattA)),"V");
    _setPropertyValue("I_BattA",
                   QString::fromStdString(std::to_string(pw.IbattA)),"A");
    _setPropertyValue("V_BattB",
                   QString::fromStdString(std::to_string(pw.VbattB)),"V");
    _setPropertyValue("I_BattB",
                   QString::fromStdString(std::to_string(pw.IbattB)),"A");
    _setPropertyValue("V_Gen",
                   QString::fromStdString(std::to_string(pw.Vgen)),"V");
    _setPropertyValue("PW_Vavionics",
                   QString::fromStdString(std::to_string(pw.Vavionics)),"V");
    _setPropertyValue("PW_Iavionics",
                   QString::fromStdString(std::to_string(pw.Iavionics)),"A");
    _setPropertyValue("PW_Vpayload",
                   QString::fromStdString(std::to_string(pw.Vpayload)),"V");
    _setPropertyValue("PW_Ipayload",
                   QString::fromStdString(std::to_string(pw.Ipayload)),"A");
    _setPropertyValue("PW_Vservo",
                   QString::fromStdString(std::to_string(pw.Vservo)),"V");
    _setPropertyValue("PW_Iservo",
                   QString::fromStdString(std::to_string(pw.Iservo)),"A");
    _setPropertyValue("PW_V28DC",
                   QString::fromStdString(std::to_string(pw.V28DC)),"V");
    _setPropertyValue("PW_I28DC",
                   QString::fromStdString(std::to_string(pw.I28DC)),"A");
    _setPropertyValue("PW_energyA",
                   QString::fromStdString(std::to_string(pw.energyA)),"mAh");
    _setPropertyValue("PW_energyB",
                   QString::fromStdString(std::to_string(pw.energyB)),"mAh");
    _setPropertyValue("PMU_Temp",
                   QString::fromStdString(std::to_string(pw.pw_temp)),"°C");

    int genstt=0;
    if(pw.Vgen > 22.4 && pw.Vgen < 72 && pw.IbattA >= 0 && pw.IbattB >= 0)
        genstt = 2;
    else if(pw.Vgen <= 22.4 && landed())
        genstt = 1;
    else
        genstt = 0;

    _setPropertyValue("GenStatus",
                   QString::fromStdString(std::to_string(genstt))," ");

//    _vBattA = pw.VbattA;
//    _vBattB = pw.VbattB;
//    _iBattA = pw.IbattA;
//    _iBattB = pw.IbattB;
//    _pwTemp = static_cast<float>(pw.pw_temp);
//    _genStatus = genstt;

//    Q_EMIT vBattAChanged();
//    Q_EMIT iBattAChanged();
//    Q_EMIT vBattBChanged();
//    Q_EMIT iBattBChanged();
//    Q_EMIT pwTempChanged();
//    Q_EMIT genStatusChanged();

    }

void Vehicle::_handleECU(mavlink_message_t &message)
{
    mavlink_ecu_t ecu;
    mavlink_msg_ecu_decode(&message, &ecu);
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif

//    _fuelUsed = static_cast<float>(ecu.fuelUsed)  / 750;

    _setPropertyValue("ECU_Throttle",
                   QString::fromStdString(std::to_string(ecu.throttle)),"%");
    _setPropertyValue("ECU_FuelUsed",
                   QString::fromStdString(std::to_string(ecu.fuelUsed)),"l");
//    _setPropertyValue("ECU_FuelUsed",
//                   QString::fromStdString(std::to_string(_fuelUsed)),"l");
    _setPropertyValue("ECU_CHT",
                   QString::fromStdString(std::to_string(ecu.CHT)),"°C");
    _setPropertyValue("ECU_FuelPressure",
                   QString::fromStdString(std::to_string(ecu.fuelPressure)),"Bar");
    _setPropertyValue("ECU_Hobbs",
                   QString::fromStdString(std::to_string(ecu.hobbs)),"s");
    _setPropertyValue("ECU_CPULoad",
                   QString::fromStdString(std::to_string(ecu.cpuLoad)),"%");
    _setPropertyValue("ECU_ChargeTemp",
                   QString::fromStdString(std::to_string(ecu.chargeTemp)),"°C");
    _setPropertyValue("ECU_FlowRate",
                   QString::fromStdString(std::to_string(ecu.flowRate))," ");
    _setPropertyValue("ECU_Rpm",
                   QString::fromStdString(std::to_string(ecu.rpm)),"RPM");
    _setPropertyValue("ECU_ThrottlePulse",
                   QString::fromStdString(std::to_string(ecu.throttlePulse))," ");

//    _engineFuelUsed = ecu.fuelUsed;
//    _engineCht = ecu.CHT;
//    _engineRpm = ecu.rpm;
//    _engineFuelPressure = ecu.fuelPressure;

//    Q_EMIT engineFuelUsedChanged();
//    Q_EMIT engineChtChanged();
//    Q_EMIT engineRpmChanged();
//    Q_EMIT engineFuelPressureChanged();
}


void Vehicle::_handleAUX_ADC(mavlink_message_t &message)
{
    mavlink_aux_adc_t aux_adc;
    mavlink_msg_aux_adc_decode(&message, &aux_adc);
#ifdef DEBUG_FUNC
    printf("%s \r\n", __func__);
#endif
    _setPropertyValue("ADC_FuelLevel",
                   QString::fromStdString(std::to_string(aux_adc.Fuel_level))," ");
    _setPropertyValue("ADC_RawFuelLevel",
                   QString::fromStdString(std::to_string(aux_adc.Raw_fuel_level))," ");
    _setPropertyValue("ADC_EnvTemp",
                   QString::fromStdString(std::to_string(aux_adc.env_temp))," ");
    _setPropertyValue("ADC_EnvRH",
                   QString::fromStdString(std::to_string(aux_adc.env_RH))," ");
    _setPropertyValue("V_Batt12S",
                   QString::fromStdString(std::to_string(aux_adc.Voltage12S_ADC))," ");

//    _vBatt12S = aux_adc.Voltage12S_ADC;
//    Q_EMIT vBatt12SChanged();
}

void Vehicle::_handleAutopilotVersion(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s\r\n", __func__);
#endif
    mavlink_autopilot_version_t autopilotVersion;
    mavlink_msg_autopilot_version_decode(&message, &autopilotVersion);
    _uid = (quint64)autopilotVersion.uid;
    Q_EMIT vehicleUIDChanged();

    if (autopilotVersion.flight_sw_version != 0) {
        int majorVersion, minorVersion, patchVersion;
        FIRMWARE_VERSION_TYPE versionType;
        majorVersion = (autopilotVersion.flight_sw_version >> (8 * 3)) & 0xFF;
        minorVersion = (autopilotVersion.flight_sw_version >> (8 * 2)) & 0xFF;
        patchVersion = (autopilotVersion.flight_sw_version >> (8 * 1)) & 0xFF;
        versionType = (FIRMWARE_VERSION_TYPE)((autopilotVersion.flight_sw_version >> (8 * 0)) & 0xFF);
        setFirmwareVersion(majorVersion, minorVersion, patchVersion, versionType);
    }
}
void Vehicle::_handleProtocolVersion(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s\r\n", __func__);
#endif
    mavlink_protocol_version_t protoVersion;
    mavlink_msg_protocol_version_decode(&message, &protoVersion);
}
void Vehicle::_handleRangeFinder(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s\r\n", __func__);
#endif
    mavlink_rangefinder_t range;
    mavlink_msg_rangefinder_decode(&message, &range);
    _sonarRange = range.distance;
    Q_EMIT sonarRangeChanged();
    _setPropertyValue("Sonarrange",QString::fromStdString(std::to_string(range.distance)),"m");
}
void Vehicle::_handleWindCov(mavlink_message_t &message)
{
//#ifdef DEBUG_FUNC
    printf("%s\r\n", __func__);
//#endif
    mavlink_wind_cov_t wind;
    mavlink_msg_wind_cov_decode(&message, &wind);
    float direction = qRadiansToDegrees(qAtan2(wind.wind_y, wind.wind_x));
    float speed = qSqrt(qPow(wind.wind_x, 2) + qPow(wind.wind_y, 2));
    if (direction < 0) {
        direction += 360;
    }
}
void Vehicle::_handleWind(mavlink_message_t &message)
{
#ifdef DEBUG_FUNC
    printf("%s\r\n", __func__);
#endif
    mavlink_wind_t wind;
    mavlink_msg_wind_decode(&message, &wind);
    // We don't want negative wind angles
    float direction = wind.direction;

    if (direction < 0) {
        direction += 360;
    }
    _setPropertyValue("WindHeading",
                      QString::fromStdString(std::to_string(static_cast<int>(direction))),"deg");
    _setPropertyValue("WindSpeed",
                      QString::fromStdString(std::to_string(static_cast<int>(wind.speed*3.6))),"km/h");
}
void Vehicle::setArmed(bool armed)
{
    printf("setArmed = %s\r\n", armed ? "true" : "false");
    // We specifically use COMMAND_LONG:MAV_CMD_COMPONENT_ARM_DISARM since it is supported by more flight stacks.
    sendMavCommand(_defaultComponentId,
                   MAV_CMD_COMPONENT_ARM_DISARM,
                   true,    // show error if fails
                   armed ? 1.0f : 0.0f,
                   21196);
}
bool Vehicle::flightModeSetAvailable(void)
{
    //    return _firmwarePlugin->isCapable(this, FirmwarePlugin::SetFlightModeCapability);
    return true;
}
QStringList Vehicle::flightModes(void)
{
    if (m_firmwarePlugin != nullptr) {
        return m_firmwarePlugin->flightModes();
    } else
        return QStringList();
}
QStringList Vehicle::flightModesOnAir(void)
{
    if (m_firmwarePlugin != nullptr) {
        return m_firmwarePlugin->flightModesOnAir();
    } else
        return QStringList();
}
QStringList Vehicle::unhealthySensors(void) const
{
    QStringList sensorList;

    struct sensorInfo_s {
        uint32_t    bit;
        const char* sensorName;
    };

    static const sensorInfo_s rgSensorInfo[] = {
        { MAV_SYS_STATUS_SENSOR_3D_GYRO,                "Gyro" },
        { MAV_SYS_STATUS_SENSOR_3D_ACCEL,               "Accelerometer" },
        { MAV_SYS_STATUS_SENSOR_3D_MAG,                 "Magnetometer" },
        { MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE,      "Absolute pressure" },
        { MAV_SYS_STATUS_SENSOR_DIFFERENTIAL_PRESSURE,  "Differential pressure" },
        { MAV_SYS_STATUS_SENSOR_GPS,                    "GPS" },
        { MAV_SYS_STATUS_SENSOR_OPTICAL_FLOW,           "Optical flow" },
        { MAV_SYS_STATUS_SENSOR_VISION_POSITION,        "Computer vision position" },
        { MAV_SYS_STATUS_SENSOR_LASER_POSITION,         "Laser based position" },
        { MAV_SYS_STATUS_SENSOR_EXTERNAL_GROUND_TRUTH,  "External ground truth" },
        { MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL,   "Angular rate control" },
        { MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION, "Attitude stabilization" },
        { MAV_SYS_STATUS_SENSOR_YAW_POSITION,           "Yaw position" },
        { MAV_SYS_STATUS_SENSOR_Z_ALTITUDE_CONTROL,     "Z/altitude control" },
        { MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL,    "X/Y position control" },
        { MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS,          "Motor outputs / control" },
        { MAV_SYS_STATUS_SENSOR_RC_RECEIVER,            "RC receiver" },
        { MAV_SYS_STATUS_SENSOR_3D_GYRO2,               "Gyro 2" },
        { MAV_SYS_STATUS_SENSOR_3D_ACCEL2,              "Accelerometer 2" },
        { MAV_SYS_STATUS_SENSOR_3D_MAG2,                "Magnetometer 2" },
        { MAV_SYS_STATUS_GEOFENCE,                      "GeoFence" },
        { MAV_SYS_STATUS_AHRS,                          "AHRS" },
        { MAV_SYS_STATUS_TERRAIN,                       "Terrain" },
        { MAV_SYS_STATUS_REVERSE_MOTOR,                 "Motors reversed" },
        { MAV_SYS_STATUS_LOGGING,                       "Logging" },
        { MAV_SYS_STATUS_SENSOR_BATTERY,                "Battery" },
    };

    for (size_t i=0; i<sizeof(rgSensorInfo)/sizeof(sensorInfo_s); i++) {
        const sensorInfo_s* pSensorInfo = &rgSensorInfo[i];
        if ((_onboardControlSensorsEnabled & pSensorInfo->bit) && !(_onboardControlSensorsHealth & pSensorInfo->bit)) {
            sensorList << pSensorInfo->sensorName;
        }
    }

    return sensorList;
}

QString Vehicle::flightMode(void)
{
    //    return _firmwarePlugin->flightMode(_base_mode, _custom_mode);
    //    printf("flightMode _custom_mode=%d\r\n",_custom_mode);
    if (m_firmwarePlugin != nullptr) {
        return m_firmwarePlugin->flightMode(static_cast<int>(_custom_mode));
    } else {
        return "UNDEFINED";
    }
}
void Vehicle::setFlightMode(const QString &flightMode)
{
    printf("Set Flight mode to %s\r\n", flightMode.toStdString().c_str());

    if (m_firmwarePlugin == nullptr) return;

    int     base_mode;
    int    custom_mode;
    bool    flightModeValid = m_firmwarePlugin->flightModeID(flightMode, &base_mode, &custom_mode);
    //    printf("m_firmwarePlugin->flightModeID(%s) = %s\r\n",
    //           flightMode.toStdString().c_str(),
    //           flightModeValid?"true":"false");

    if (flightModeValid) {
        printf("flightModeValid \r\n");
        // setFlightMode will only set MAV_MODE_FLAG_CUSTOM_MODE_ENABLED in base_mode, we need to move back in the existing
        // states.
        uint8_t newBaseMode = _base_mode & ~MAV_MODE_FLAG_DECODE_POSITION_CUSTOM_MODE;
        newBaseMode |= base_mode;
        mavlink_message_t msg;
        mavlink_msg_set_mode_pack_chan(m_com->systemId(),
                                       m_com->componentId(),
                                       m_com->mavlinkChannel(),
                                       &msg,
                                       id(),
                                       newBaseMode,
                                       custom_mode);
        sendMessageOnLink(m_com, msg);
    } else {
        printf("flightMode not Valid \r\n");
        qWarning() << "FirmwarePlugin::setFlightMode failed, flightMode:" << flightMode;
    }
}
bool Vehicle::sendMessageOnLink(IOFlightController *link, mavlink_message_t message)
{
#ifdef DEBUG_FUNC
    printf("%s message.msgid[%d] message.compid[%d]\r\n", __func__, message.msgid, message.compid);
#endif

    // Write message into buffer, prepending start sign
    if (!link || !link->m_linkInterface->isOpen()) {
        return false;
    }

    Q_EMIT _sendMessageOnLinkOnThread(link, message);
}
void Vehicle::sendMavCommand(int component, MAV_CMD command, bool showError, float param1, float param2, float param3, float param4, float param5, float param6, float param7)
{
#ifdef DEBUG_FUNC
    printf("%s[component=%d command=%d]\r\n", __func__, component, command);
#endif
    MavCommandQueueEntry_t entry;
    entry.commandInt = false;
    entry.component = component;
    entry.command = command;
    entry.showError = showError;
    entry.rgParam[0] = param1;
    entry.rgParam[1] = param2;
    entry.rgParam[2] = param3;
    entry.rgParam[3] = param4;
    entry.rgParam[4] = param5;
    entry.rgParam[5] = param6;
    entry.rgParam[6] = param7;
    _mavCommandQueue.append(entry);

    if (_mavCommandQueue.count() == 1) {
        _mavCommandRetryCount = 0;
        _sendMavCommandAgain();
    }
}
void Vehicle::sendMavCommandInt(int component, MAV_CMD command, MAV_FRAME frame, bool showError, float param1, float param2, float param3, float param4, double param5, double param6, float param7)
{
#ifdef DEBUG_FUNC
    printf("%s[component=%d command=%d]\r\n", __func__, component, command);
#endif
    MavCommandQueueEntry_t entry;
    entry.commandInt = true;
    entry.component = component;
    entry.command = command;
    entry.frame = frame;
    entry.showError = showError;
    entry.rgParam[0] = param1;
    entry.rgParam[1] = param2;
    entry.rgParam[2] = param3;
    entry.rgParam[3] = param4;
    entry.rgParam[4] = param5;
    entry.rgParam[5] = param6;
    entry.rgParam[6] = param7;
    _mavCommandQueue.append(entry);

    if (_mavCommandQueue.count() == 1) {
        _mavCommandRetryCount = 0;
        _sendMavCommandAgain();
    }
}
void Vehicle::_sendMavCommandAgain(void)
{
    if (!_mavCommandQueue.size()) {
        qWarning() << "Command resend with no commands in queue";
        _mavCommandAckTimer.stop();
        return;
    }

    MavCommandQueueEntry_t &queuedCommand = _mavCommandQueue[0];

    if (_mavCommandRetryCount++ > _mavCommandMaxRetryCount) {
        if (queuedCommand.command == MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES) {
            // We aren't going to get a response back for capabilities, so stop waiting for it before we ask for mission items
            //            qCDebug(VehicleLog) << "Vehicle failed to responded to MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES. Setting no capabilities. Starting Plan request.";
            //            _setCapabilities(0);
            //            _startPlanRequest();
        }

        if (queuedCommand.command == MAV_CMD_REQUEST_PROTOCOL_VERSION) {
//            // We aren't going to get a response back for the protocol version, so assume v1 is all we can do.
//            // If the max protocol version is uninitialized, fall back to v1.
//            qCDebug(VehicleLog) << "Vehicle failed to responded to MAV_CMD_REQUEST_PROTOCOL_VERSION. Starting Plan request.";
//            if (_maxProtoVersion == 0) {
//                qCDebug(VehicleLog) << "Setting _maxProtoVersion to 100 since not yet set.";
//                _setMaxProtoVersion(100);
//            } else {
//                qCDebug(VehicleLog) << "Leaving _maxProtoVersion at current value" << _maxProtoVersion;
//            }
        }
        if(queuedCommand.command == MAV_CMD_DO_SET_HOME){
            printf("Failed to set home position\r\n");
            Q_EMIT homePositionChanged(_homePosition);
        }
        Q_EMIT mavCommandResult(_id, queuedCommand.component, queuedCommand.command, MAV_RESULT_FAILED, true /* noResponsefromVehicle */);
//        if (queuedCommand.showError) {
//            qgcApp()->showMessage(tr("Vehicle did not respond to command: %1").arg(_toolbox->missionCommandTree()->friendlyName(queuedCommand.command)));
//        }
        _mavCommandQueue.removeFirst();
        _sendNextQueuedMavCommand();
        return;
    }

    if (_mavCommandRetryCount > 1) {
        // We always let AUTOPILOT_CAPABILITIES go through multiple times even if we don't get acks. This is because
        // we really need to get capabilities and version info back over a lossy link.
        //        if (queuedCommand.command != MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES) {
        //            if (px4Firmware()) {
        //                // Older PX4 firmwares are inconsistent with repect to sending back an Ack from a COMMAND_LONG, hence we can't support retry logic for it.
        //                if (_firmwareMajorVersion != versionNotSetValue) {
        //                    // If no version set assume lastest master dev build, so acks are suppored
        //                    if (_firmwareMajorVersion <= 1 && _firmwareMinorVersion <= 5 && _firmwarePatchVersion <= 3) {
        //                        // Acks not supported in this version
        //                        return;
        //                    }
        //                }
        //            } else {
        //                if (queuedCommand.command == MAV_CMD_START_RX_PAIR) {
        //                    // The implementation of this command comes from the IO layer and is shared across stacks. So for other firmwares
        //                    // we aren't really sure whether they are correct or not.
        //                    return;
        //                }
        //            }
        //        }
        //        qCDebug(VehicleLog) << "Vehicle::_sendMavCommandAgain retrying command:_mavCommandRetryCount" << queuedCommand.command << _mavCommandRetryCount;
    }

    _mavCommandAckTimer.start();
    mavlink_message_t       msg;

    if (queuedCommand.commandInt) {
        mavlink_command_int_t  cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.target_system =     _id;
        cmd.target_component =  queuedCommand.component;
        cmd.command =           queuedCommand.command;
        cmd.frame =             queuedCommand.frame;
        cmd.param1 =            queuedCommand.rgParam[0];
        cmd.param2 =            queuedCommand.rgParam[1];
        cmd.param3 =            queuedCommand.rgParam[2];
        cmd.param4 =            queuedCommand.rgParam[3];
        cmd.x =                 queuedCommand.rgParam[4] * qPow(10.0, 7.0);
        cmd.y =                 queuedCommand.rgParam[5] * qPow(10.0, 7.0);
        cmd.z =                 queuedCommand.rgParam[6];
        mavlink_msg_command_int_encode_chan(m_com->systemId(),
                                            m_com->componentId(),
                                            m_com->mavlinkChannel(),
                                            &msg,
                                            &cmd);
    } else {
        mavlink_command_long_t  cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.target_system =     _id;
        cmd.target_component =  queuedCommand.component;
        cmd.command =           queuedCommand.command;
        cmd.confirmation =      0;
        cmd.param1 =            queuedCommand.rgParam[0];
        cmd.param2 =            queuedCommand.rgParam[1];
        cmd.param3 =            queuedCommand.rgParam[2];
        cmd.param4 =            queuedCommand.rgParam[3];
        cmd.param5 =            queuedCommand.rgParam[4];
        cmd.param6 =            queuedCommand.rgParam[5];
        cmd.param7 =            queuedCommand.rgParam[6];
        mavlink_msg_command_long_encode_chan(m_com->systemId(),
                                             m_com->componentId(),
                                             m_com->mavlinkChannel(),
                                             &msg,
                                             &cmd);
    }

    sendMessageOnLink(m_com, msg);
}
void Vehicle::_sendNextQueuedMavCommand(void)
{
    if (_mavCommandQueue.count()) {
        _mavCommandRetryCount = 0;
        _sendMavCommandAgain();
    }
}
void Vehicle::_handleCommandAck(mavlink_message_t &message)
{
    bool showError = false;
    mavlink_command_ack_t ack;
    mavlink_msg_command_ack_decode(&message, &ack);
    printf("%s ack.command[%d]\r\n", __func__, ack.command);
    if (ack.command == MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES && ack.result != MAV_RESULT_ACCEPTED) {
        // We aren't going to get a response back for capabilities, so stop waiting for it before we ask for mission items
        //        qCDebug(VehicleLog) << QStringLiteral("Vehicle responded to MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES with error(%1). Setting no capabilities. Starting Plan request.").arg(ack.result);
        //        _setCapabilities(0);
    }

    if (ack.command == MAV_CMD_REQUEST_PROTOCOL_VERSION && ack.result != MAV_RESULT_ACCEPTED) {
        // The autopilot does not understand the request and consequently is likely handling only
        // MAVLink 1
        //        qCDebug(VehicleLog) << QStringLiteral("Vehicle responded to MAV_CMD_REQUEST_PROTOCOL_VERSION with error(%1).").arg(ack.result);
        //        if (_maxProtoVersion == 0) {
        //            qCDebug(VehicleLog) << "Setting _maxProtoVersion to 100 since not yet set.";
        //            _setMaxProtoVersion(100);
        //        } else {
        //            qCDebug(VehicleLog) << "Leaving _maxProtoVersion at current value" << _maxProtoVersion;
        //        }
        // FIXME: Is this missing here. I believe it is a bug. Debug to verify. May need to go into Stable.
        //_startPlanRequest();
    }

    if (_mavCommandQueue.count() && ack.command == _mavCommandQueue[0].command) {
        _mavCommandAckTimer.stop();
        showError = _mavCommandQueue[0].showError;
        _mavCommandQueue.removeFirst();
    }

    mavCommandResult(_id, message.compid, ack.command, ack.result, false /* noResponsefromVehicle */);
    //    if (showError) {
    //        QString commandName = _toolbox->missionCommandTree()->friendlyName((MAV_CMD)ack.command);
    //        switch (ack.result) {
    //        case MAV_RESULT_TEMPORARILY_REJECTED:
    //            qgcApp()->showMessage(tr("%1 command temporarily rejected").arg(commandName));
    //            break;
    //        case MAV_RESULT_DENIED:
    //            qgcApp()->showMessage(tr("%1 command denied").arg(commandName));
    //            break;
    //        case MAV_RESULT_UNSUPPORTED:
    //            qgcApp()->showMessage(tr("%1 command not supported").arg(commandName));
    //            break;
    //        case MAV_RESULT_FAILED:
    //            qgcApp()->showMessage(tr("%1 command failed").arg(commandName));
    //            break;
    //        default:
    //            // Do nothing
    //            break;
    //        }
    //    }
    _sendNextQueuedMavCommand();
}
