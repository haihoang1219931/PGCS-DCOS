#pragma once
#ifndef VEHICLE_H
#define VEHICLE_H

#include <QObject>
#include <QDateTime>
#include <QTimer>
#include <QMap>
#include <QGeoCoordinate>
#include <qmath.h>
#include <QVariant>
#include <QQmlListProperty>
#include "../Com/IOFlightController.h"
#include "../Mission/PlanController.h"
#include "../Params/ParamsController.h"
#include "../Params/Fact.h"
#include "../UAS/UAS.h"
#include "../Com/QGCMAVLink.h"
#include <ardupilotmega/ardupilotmega.h>
#include "../../Joystick/Files/FileControler.h"
//#define DEBUG
//#define DEBUG_FUNC
//#define DEBUG_MESSAGE_RECV
class UAS;
class ParamsController;
class PlanController;
class IOFlightController;
class FirmwarePlugin;
class FirmwarePluginManager;

class Vehicle : public QObject
{
    Q_OBJECT
    Q_PROPERTY(Vehicle*             uav                         READ uav            WRITE setUav)
    Q_PROPERTY(IOFlightController*  communication               READ communication      WRITE setCommunication)
    Q_PROPERTY(PlanController*      planController              READ planController     WRITE setPlanController)
    Q_PROPERTY(ParamsController*    paramsController            READ paramsController   WRITE setParamsController)
    Q_PROPERTY(QStringList          flightModes                 READ flightModes                                    NOTIFY flightModesChanged)
    Q_PROPERTY(QString              flightMode                  READ flightMode         WRITE setFlightMode         NOTIFY flightModeChanged)
    Q_PROPERTY(bool                 armed                       READ armed                                          NOTIFY armedChanged)
    Q_PROPERTY(bool                 landed                      READ landed                                         NOTIFY landedChanged)
    Q_PROPERTY(QGeoCoordinate       coordinate                  READ coordinate                                     NOTIFY coordinateChanged)
    Q_PROPERTY(QGeoCoordinate       homePosition                READ homePosition       WRITE setHomePosition       NOTIFY homePositionChanged)
    Q_PROPERTY(float                roll                        READ roll                                           NOTIFY rollChanged)
    Q_PROPERTY(float                pitch                       READ pitch                                          NOTIFY pitchChanged)
    Q_PROPERTY(float                heading                     READ heading                                        NOTIFY headingChanged)
    Q_PROPERTY(float                airSpeed                    READ airSpeed                                       NOTIFY airSpeedChanged)
    Q_PROPERTY(float                altitudeRelative            READ altitudeRelative                               NOTIFY altitudeRelativeChanged)
    Q_PROPERTY(float                engineSensor_1              READ engineSensor_1                                 NOTIFY engineSensor_1Changed)
    Q_PROPERTY(float                engineSensor_2              READ engineSensor_2                                 NOTIFY engineSensor_2Changed)
    Q_PROPERTY(bool                 gpsSignal                   READ gpsSignal                                      NOTIFY gpsChanged)
    Q_PROPERTY(bool                 link                        READ link               WRITE setLink               NOTIFY linkChanged)
    Q_PROPERTY(QString              ekfSignal                   READ ekfSignal          WRITE setEkfSignal          NOTIFY ekfChanged)
    Q_PROPERTY(QString              vibeSignal                  READ vibeSignal         WRITE setVibeSignal         NOTIFY vibeChanged)
    Q_PROPERTY(float                headingToHome               READ headingToHome                                  NOTIFY headingToHomeChanged)
    Q_PROPERTY(float                distanceToHome              READ distanceToHome                                 NOTIFY distanceToHomeChanged)
    Q_PROPERTY(int                currentWaypoint             READ currentWaypoint                                NOTIFY currentWaypointChanged)
    Q_PROPERTY(float                distanceToCurrentWaypoint   READ distanceToCurrentWaypoint            NOTIFY distanceToCurrentWaypointChanged)
    Q_PROPERTY(float                batteryVoltage              READ batteryVoltage                                 NOTIFY batteryVoltageChanged)
    Q_PROPERTY(float                batteryAmpe                 READ batteryAmpe                                    NOTIFY batteryAmpeChanged)
    Q_PROPERTY(float                groundSpeed                 READ groundSpeed                                    NOTIFY groundSpeedChanged)
    Q_PROPERTY(float                climbSpeed                  READ climbSpeed                                     NOTIFY climbSpeedChanged)
    Q_PROPERTY(float                altitudeAMSL                READ altitudeAMSL                                   NOTIFY altitudeAMSLChanged)
    Q_PROPERTY(float                altitudeAGL                 READ altitudeAGL                                    NOTIFY altitudeAGLChanged)
    Q_PROPERTY(float                latGPS                      READ latGPS                                         NOTIFY latGPSChanged)
    Q_PROPERTY(float                lonGPS                      READ lonGPS                                         NOTIFY lonGPSChanged)
    Q_PROPERTY(float                hdopGPS                     READ hdopGPS                                        NOTIFY hdopGPSChanged)
    Q_PROPERTY(float                vdopGPS                     READ vdopGPS                                        NOTIFY vdopGPSChanged)
    Q_PROPERTY(float                courseOverGroundGPS         READ courseOverGroundGPS                            NOTIFY courseOverGroundGPSChanged)
    Q_PROPERTY(int                  countGPS                    READ countGPS                                       NOTIFY countGPSChanged)
    Q_PROPERTY(QString              lockGPS                     READ lockGPS                                        NOTIFY lockGPSChanged)
    Q_PROPERTY(QString              messageSecurity             READ messageSecurity    WRITE setMessageSecurity    NOTIFY messageSecurityChanged)
    Q_PROPERTY(UAS*                 uas                         READ uas                                            NOTIFY uasChanged)
    Q_PROPERTY(QStringList          unhealthySensors            READ unhealthySensors                               NOTIFY unhealthySensorsChanged)
    Q_PROPERTY(int                  sensorsPresentBits          READ sensorsPresentBits                             NOTIFY sensorsPresentBitsChanged)
    Q_PROPERTY(int                  sensorsEnabledBits          READ sensorsEnabledBits                             NOTIFY sensorsEnabledBitsChanged)
    Q_PROPERTY(int                  sensorsHealthBits           READ sensorsHealthBits                              NOTIFY sensorsHealthBitsChanged)
    Q_PROPERTY(quint64              mavlinkSentCount            READ mavlinkSentCount                               NOTIFY mavlinkStatusChanged)
    Q_PROPERTY(quint64              mavlinkReceivedCount        READ mavlinkReceivedCount                           NOTIFY mavlinkStatusChanged)
    Q_PROPERTY(quint64              mavlinkLossCount            READ mavlinkLossCount                               NOTIFY mavlinkStatusChanged)
    Q_PROPERTY(float                mavlinkLossPercent          READ mavlinkLossPercent                             NOTIFY mavlinkStatusChanged)
    Q_PROPERTY(VEHICLE_MAV_TYPE     vehicleType                 READ vehicleType        WRITE setVehicleType        NOTIFY vehicleTypeChanged)
    Q_PROPERTY(float                paramAirSpeed               READ paramAirSpeed      WRITE setParamAirSpeed      NOTIFY paramAirSpeedChanged)
    Q_PROPERTY(float                paramLoiterRadius           READ paramLoiterRadius      WRITE setParamLoiterRadius      NOTIFY paramLoiterRadiusChanged)
    Q_PROPERTY(int                  rssi                        READ rssi                                           NOTIFY rssiChanged)
    Q_PROPERTY(float                pressABS                    READ pressABS                                       NOTIFY pressABSChanged)
    Q_PROPERTY(int                  temperature                 READ temperature                                    NOTIFY temperatureChanged)
    Q_PROPERTY( QQmlListProperty<Fact> propertiesModel          READ propertiesModel                                     NOTIFY propertiesModelChanged)
    Q_PROPERTY(int                  propertiesShowCount         READ propertiesShowCount    WRITE setPropertiesShowCount    NOTIFY propertiesShowCountChanged)
    Q_PROPERTY( QQmlListProperty<Fact> paramsModel              READ paramsModel                                    NOTIFY paramsModelChanged)
public:
    typedef enum {
        MessageNone,
        MessageNormal,
        MessageWarning,
        MessageError
    } MessageType_t;
    typedef struct {
        int         component;
        bool        commandInt; // true: use COMMAND_INT, false: use COMMAND_LONG
        MAV_CMD     command;
        MAV_FRAME   frame;
        double      rgParam[7];
        bool        showError;
    } MavCommandQueueEntry_t;

    explicit Vehicle(QObject *parent = nullptr);
    ~Vehicle();
    enum MavlinkSysStatus {
        SysStatusSensor3dGyro =                 MAV_SYS_STATUS_SENSOR_3D_GYRO,
        SysStatusSensor3dAccel =                MAV_SYS_STATUS_SENSOR_3D_ACCEL,
        SysStatusSensor3dMag =                  MAV_SYS_STATUS_SENSOR_3D_MAG,
        SysStatusSensorAbsolutePressure =       MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE,
        SysStatusSensorDifferentialPressure =   MAV_SYS_STATUS_SENSOR_DIFFERENTIAL_PRESSURE,
        SysStatusSensorGPS =                    MAV_SYS_STATUS_SENSOR_GPS,
        SysStatusSensorOpticalFlow =            MAV_SYS_STATUS_SENSOR_OPTICAL_FLOW,
        SysStatusSensorVisionPosition =         MAV_SYS_STATUS_SENSOR_VISION_POSITION,
        SysStatusSensorLaserPosition =          MAV_SYS_STATUS_SENSOR_LASER_POSITION,
        SysStatusSensorExternalGroundTruth =    MAV_SYS_STATUS_SENSOR_EXTERNAL_GROUND_TRUTH,
        SysStatusSensorAngularRateControl =     MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL,
        SysStatusSensorAttitudeStabilization =  MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION,
        SysStatusSensorYawPosition =            MAV_SYS_STATUS_SENSOR_YAW_POSITION,
        SysStatusSensorZAltitudeControl =       MAV_SYS_STATUS_SENSOR_Z_ALTITUDE_CONTROL,
        SysStatusSensorXYPositionControl =      MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL,
        SysStatusSensorMotorOutputs =           MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS,
        SysStatusSensorRCReceiver =             MAV_SYS_STATUS_SENSOR_RC_RECEIVER,
        SysStatusSensor3dGyro2 =                MAV_SYS_STATUS_SENSOR_3D_GYRO2,
        SysStatusSensor3dAccel2 =               MAV_SYS_STATUS_SENSOR_3D_ACCEL2,
        SysStatusSensor3dMag2 =                 MAV_SYS_STATUS_SENSOR_3D_MAG2,
        SysStatusSensorGeoFence =               MAV_SYS_STATUS_GEOFENCE,
        SysStatusSensorAHRS =                   MAV_SYS_STATUS_AHRS,
        SysStatusSensorTerrain =                MAV_SYS_STATUS_TERRAIN,
        SysStatusSensorReverseMotor =           MAV_SYS_STATUS_REVERSE_MOTOR,
        SysStatusSensorLogging =                MAV_SYS_STATUS_LOGGING,
        SysStatusSensorBattery =                MAV_SYS_STATUS_SENSOR_BATTERY,
    };
    Q_ENUM(MavlinkSysStatus)
    enum VEHICLE_MAV_TYPE
    {
       MAV_TYPE_GENERIC=0, /* Generic micro air vehicle. | */
       MAV_TYPE_FIXED_WING=1, /* Fixed wing aircraft. | */
       MAV_TYPE_QUADROTOR=2, /* Quadrotor | */
       MAV_TYPE_COAXIAL=3, /* Coaxial helicopter | */
       MAV_TYPE_HELICOPTER=4, /* Normal helicopter with tail rotor. | */
       MAV_TYPE_ANTENNA_TRACKER=5, /* Ground installation | */
       MAV_TYPE_GCS=6, /* Operator control unit / ground control station | */
       MAV_TYPE_AIRSHIP=7, /* Airship, controlled | */
       MAV_TYPE_FREE_BALLOON=8, /* Free balloon, uncontrolled | */
       MAV_TYPE_ROCKET=9, /* Rocket | */
       MAV_TYPE_GROUND_ROVER=10, /* Ground rover | */
       MAV_TYPE_SURFACE_BOAT=11, /* Surface vessel, boat, ship | */
       MAV_TYPE_SUBMARINE=12, /* Submarine | */
       MAV_TYPE_HEXAROTOR=13, /* Hexarotor | */
       MAV_TYPE_OCTOROTOR=14, /* Octorotor | */
       MAV_TYPE_TRICOPTER=15, /* Tricopter | */
       MAV_TYPE_FLAPPING_WING=16, /* Flapping wing | */
       MAV_TYPE_KITE=17, /* Kite | */
       MAV_TYPE_ONBOARD_CONTROLLER=18, /* Onboard companion controller | */
       MAV_TYPE_VTOL_DUOROTOR=19, /* Two-rotor VTOL using control surfaces in vertical operation in addition. Tailsitter. | */
       MAV_TYPE_VTOL_QUADROTOR=20, /* Quad-rotor VTOL using a V-shaped quad config in vertical operation. Tailsitter. | */
       MAV_TYPE_VTOL_TILTROTOR=21, /* Tiltrotor VTOL | */
       MAV_TYPE_VTOL_RESERVED2=22, /* VTOL reserved 2 | */
       MAV_TYPE_VTOL_RESERVED3=23, /* VTOL reserved 3 | */
       MAV_TYPE_VTOL_RESERVED4=24, /* VTOL reserved 4 | */
       MAV_TYPE_VTOL_RESERVED5=25, /* VTOL reserved 5 | */
       MAV_TYPE_GIMBAL=26, /* Gimbal (standalone) | */
       MAV_TYPE_ADSB=27, /* ADSB system (standalone) | */
       MAV_TYPE_PARAFOIL=28, /* Steerable, nonrigid airfoil | */
       MAV_TYPE_DODECAROTOR=29, /* Dodecarotor | */
       MAV_TYPE_CAMERA=30, /* Camera (standalone) | */
       MAV_TYPE_CHARGING_STATION=31, /* Charging station | */
       MAV_TYPE_FLARM=32, /* FLARM collision avoidance system (standalone) | */
       MAV_TYPE_ENUM_END=33, /*  | */
    };
    Q_ENUMS(VEHICLE_MAV_TYPE)
public:
    int defaultComponentId(){return _defaultComponentId;}
    float roll(){return _roll;}
    float pitch(){return _pitch;}
    float heading(){return _heading;}
    float airSpeed(){return _airSpeed;}
    float altitudeRelative(){return _altitudeRelative;}
    float engineSensor_1(){return _engineSensor_1;}
    float engineSensor_2(){return _engineSensor_2;}
    QGeoCoordinate coordinate(){ return _coordinate;}

    bool gpsSignal(){ return _gpsFixedType > 1;}
    bool link(){ return _link;}
    void setLink(bool link){
        printf("Link to %s change to %s\r\n",m_uav==nullptr?"UAV":"Tracker", link?"ON":"OFF");
        if(_link != link){
            _link = link;
            Q_EMIT linkChanged();
        }
    }
    QString ekfSignal(){ return _ekf;}
    QString vibeSignal(){ return _vibration;}
    void setEkfSignal(QString ekf){
        if(ekf!=_ekf){
            _ekf = ekf;
            Q_EMIT ekfChanged();
        }
    }
    void setVibeSignal(QString vibe){
        if(vibe!=_vibration){
            _vibration = vibe;
            Q_EMIT vibeChanged();
        }
    }
    float headingToHome(){ return _headingToHome;}
    float distanceToHome(){ return _distanceToHome;}
    int currentWaypoint(){ return _currentWaypoint;}
    void setCurrentWaypoint(int currentWaypoint){
        _currentWaypoint = currentWaypoint;
        Q_EMIT currentWaypointChanged();
    }
    float distanceToCurrentWaypoint(){ return _distanceToCurrentWaypoint;}
    void setDistanceToCurrentWaypoint(int distanceToCurrentWaypoint){
        _distanceToCurrentWaypoint = distanceToCurrentWaypoint;
        _setPropertyValue("DisttoWP",
                                 QString::fromStdString(std::to_string(_distanceToCurrentWaypoint)),
                                 "m");
        Q_EMIT distanceToCurrentWaypointChanged();
    }
    QGeoCoordinate homePosition(void){return _homePosition;}
    void setHomePosition(QGeoCoordinate homePosition){
        _homePosition = homePosition;
        _setPropertyValue("AltHome",QString::fromStdString(std::to_string(_homePosition.altitude())),"m");
        Q_EMIT homePositionChanged(_homePosition);
    }
    float batteryVoltage(){ return _batteryVoltage;}
    float batteryAmpe(){ return _batteryAmpe;}
    float groundSpeed(){ return _groundSpeed;}
    float climbSpeed(){ return _climbSpeed;}
    float altitudeAMSL(){return _altitudeAMSL;}
    float altitudeAGL(){ return _altitudeAGL;}
    float latGPS(){ return _latGPS;}
    float lonGPS(){ return _lonGPS;}
    float hdopGPS(){ return _hdopGPS;}
    float vdopGPS(){ return _vdopGPS;}
    float courseOverGroundGPS(){ return _courseOverGroundGPS;}
    int   countGPS(){ return _countGPS;}
    QString lockGPS(){ return _lockGPS;}
    QString messageSecurity(){ return _messageSecurity;}
    void setMessageSecurity(QString messageSecurity){
        if(_messageSecurity != messageSecurity){
            _messageSecurity = messageSecurity;
            Q_EMIT messageSecurityChanged();
        }
    }
    UAS* uas(){ return m_uas;}
    QStringList     unhealthySensors        () const;
    int             sensorsPresentBits      () const { return _onboardControlSensorsPresent; }
    int             sensorsEnabledBits      () const { return _onboardControlSensorsEnabled; }
    int             sensorsHealthBits       () const { return _onboardControlSensorsHealth; }
    int             sensorsUnhealthyBits    () const { return _onboardControlSensorsUnhealthy; }
    bool            px4Firmware             () const { return _firmwareType == MAV_AUTOPILOT_PX4; }
    bool            apmFirmware             () const { return _firmwareType == MAV_AUTOPILOT_ARDUPILOTMEGA; }
    bool            genericFirmware         () const { return !px4Firmware() && !apmFirmware(); }

    quint64     mavlinkSentCount        () { return _mavlinkSentCount; }        /// Calculated total number of messages sent to us
    quint64     mavlinkReceivedCount    () { return _mavlinkReceivedCount; }    /// Total number of sucessful messages received
    quint64     mavlinkLossCount        () { return _mavlinkLossCount; }        /// Total number of lost messages
    float       mavlinkLossPercent      () { return _mavlinkLossPercent; }      /// Running loss rate
    int rssi(){ return _telemetryLRSSI; }
    float pressABS(){ return _pressABS; }
    int temperature(){ return _temperature; }
public:
    /// Command vehicle to change loiter time
    Q_INVOKABLE void commandLoiterRadius(float radius);
    /// Command vehicle to return to launch
    Q_INVOKABLE void commandRTL(void);

    /// Command vehicle to land at current location
    Q_INVOKABLE void commandLand(void);

    /// Command vehicle to takeoff from current location
    Q_INVOKABLE void commandTakeoff(double altitudeRelative);

    /// @return The minimum takeoff altitude (relative) for guided takeoff.
    Q_INVOKABLE double minimumTakeoffAltitude(void);

    /// Command vehicle to move to specified location (altitude is included and relative)
    Q_INVOKABLE void commandGotoLocation(const QGeoCoordinate& gotoCoord);

    /// Command vehicle to change altitude
    ///     @param altitudeChange If > 0, go up by amount specified, if < 0, go down by amount specified
    Q_INVOKABLE void commandChangeAltitude(double altitudeChange);
    /// Command vehicle to change altitude
    ///     @param altitudeChange If > 0, go up by amount specified, if < 0, go down by amount specified
    Q_INVOKABLE void commandSetAltitude(double newAltitude);
    /// Command vehicle to change speed
    ///     @param speedChange If > 0, go up by amount specified, if < 0, go down by amount specified
    Q_INVOKABLE void commandChangeSpeed(double speedChange);
    /// Command vehicle to orbit given center point
    ///     @param centerCoord Orit around this point
    ///     @param radius Distance from vehicle to centerCoord
    ///     @param amslAltitude Desired vehicle altitude
    Q_INVOKABLE void commandOrbit(const QGeoCoordinate& centerCoord, double radius, double amslAltitude);

    /// Command vehicle to pause at current location. If vehicle supports guide mode, vehicle will be left
    /// in guided mode after pause.
    Q_INVOKABLE void pauseVehicle(void);

    /// Command vehicle to kill all motors no matter what state
    Q_INVOKABLE void emergencyStop(void);

    /// Command vehicle to abort landing
    Q_INVOKABLE void abortLanding(double climbOutAltitude);

    Q_INVOKABLE void startMission(void);

    /// Alter the current mission item on the vehicle
    Q_INVOKABLE void setCurrentMissionSequence(int seq);

    /// Reboot vehicle
    Q_INVOKABLE void rebootVehicle();

    /// Clear Messages
    Q_INVOKABLE void clearMessages();

    Q_INVOKABLE void triggerCamera(void);
    Q_INVOKABLE void sendPlan(QString planFile);

    /// Used to check if running current version is equal or higher than the one being compared.
    //  returns 1 if current > compare, 0 if current == compare, -1 if current < compare
    Q_INVOKABLE int versionCompare(QString& compare);
    Q_INVOKABLE int versionCompare(int major, int minor, int patch);

    /// Test motor
    ///     @param motor Motor number, 1-based
    ///     @param percent 0-no power, 100-full power
    Q_INVOKABLE void motorTest(int motor, int percent);

    /// Set home postion
    ///     @param lat: latitude
    ///     @param lon: longitude
    Q_INVOKABLE void setHomeLocation(float lat, float lon);

    /// Set rtl altitude via param
    ///     @param lat: latitude
    ///     @param lon: longitude
    ///     @param alt: altitude
    Q_INVOKABLE void setAltitudeRTL(float alt);

    Q_INVOKABLE void sendHomePosition(QGeoCoordinate location);

public:
    Q_INVOKABLE void activeProperty(QString name,bool active);
    Q_INVOKABLE int countActiveProperties();
public:
    Vehicle* uav();
    void setUav(Vehicle* uav);
    ParamsController* paramsController();
    void setParamsController(ParamsController* paramsController);
    PlanController* planController();
    void setPlanController(PlanController* planController);
    IOFlightController* communication();
    void setCommunication(IOFlightController* com);
    ParamsController* params();
    bool armed(void) { return _armed; }
    Q_INVOKABLE void setArmed(bool armed);

    bool landed(void) { return _landed; }

    bool flightModeSetAvailable(void);
    QStringList flightModes(void);
    QString flightMode(void);
    void setFlightMode(const QString& flightMode);

    /// Q_SIGNALS: mavCommandResult on success or failure
    void sendMavCommand(int component, MAV_CMD command, bool showError, float param1 = 0.0f, float param2 = 0.0f, float param3 = 0.0f, float param4 = 0.0f, float param5 = 0.0f, float param6 = 0.0f, float param7 = 0.0f);
    void sendMavCommandInt(int component, MAV_CMD command, MAV_FRAME frame, bool showError, float param1, float param2, float param3, float param4, double param5, double param6, float param7);

    /// Same as sendMavCommand but available from Qml.
    Q_INVOKABLE void sendCommand(int component, int command, bool showError, double param1 = 0.0, double param2 = 0.0, double param3 = 0.0, double param4 = 0.0, double param5 = 0.0, double param6 = 0.0, double param7 = 0.0)
    {
        sendMavCommand(
            component, static_cast<MAV_CMD>(command),
            showError,
            static_cast<float>(param1),
            static_cast<float>(param2),
            static_cast<float>(param3),
            static_cast<float>(param4),
            static_cast<float>(param5),
            static_cast<float>(param6),
            static_cast<float>(param7));
    }

    int firmwareMajorVersion(void) const { return _firmwareMajorVersion; }
    int firmwareMinorVersion(void) const { return _firmwareMinorVersion; }
    int firmwarePatchVersion(void) const { return _firmwarePatchVersion; }
    int firmwareVersionType(void) const { return _firmwareVersionType; }
    int firmwareCustomMajorVersion(void) const { return _firmwareCustomMajorVersion; }
    int firmwareCustomMinorVersion(void) const { return _firmwareCustomMinorVersion; }
    int firmwareCustomPatchVersion(void) const { return _firmwareCustomPatchVersion; }
    QString firmwareVersionTypeString(void) const;
    void setFirmwareVersion(int majorVersion, int minorVersion, int patchVersion, FIRMWARE_VERSION_TYPE versionType = FIRMWARE_VERSION_TYPE_OFFICIAL);
    void setFirmwareCustomVersion(int majorVersion, int minorVersion, int patchVersion);
    // Property accesors
    int id(void) { return _id; }
    MAV_AUTOPILOT firmwareType(void) const { return _firmwareType; }
    VEHICLE_MAV_TYPE vehicleType(void) const { return _vehicleType; }
    void setVehicleType(VEHICLE_MAV_TYPE vehicleType){
        if(_vehicleType != vehicleType){
            _vehicleType = vehicleType;
            printf("_vehicleType = %d\r\n",_vehicleType);
            Q_EMIT vehicleTypeChanged();
        }
    }


    float paramLoiterRadius(){ return _paramLoiterRadius;}
    void setParamLoiterRadius(float value){
        _paramLoiterRadius = value;
        Q_EMIT paramLoiterRadiusChanged();
    }
    float paramAirSpeed(){ return _paramAirSpeed;}
    void setParamAirSpeed(float speed){
        _paramAirSpeed = speed;
        Q_EMIT paramAirSpeedChanged();
    }
    void setPropertiesShowCount(int value){
        if(value >=0 && value<= _propertiesModel.size() && _propertiesShowCount != value){
            _propertiesShowCount = value;
            Q_EMIT propertiesShowCountChanged();
        }
    }
    int propertiesShowCount(){ return _propertiesShowCount;}
    QQmlListProperty<Fact> propertiesModel()
    {
        return QQmlListProperty<Fact>(this, _propertiesModel);
    }
    QQmlListProperty<Fact> paramsModel()
    {
        return QQmlListProperty<Fact>(this, _paramsModel);
    }
    bool sendMessageOnLink(IOFlightController* link, mavlink_message_t message);
    void _sendMavCommandAgain(void);
    void _sendNextQueuedMavCommand(void);
    void setHomeAltitude(float homeAltitude){ _homeAltitude = homeAltitude;}
    void _setParamValue(QString name,QString value,QString unit,bool notify = false);
Q_SIGNALS:
    // forward signal to other component
    void missingParametersChanged(bool missingParameters);
    void loadProgressChanged(float value);
    void mavlinkMessageReceived(mavlink_message_t message);
    void mavCommandResult(int vehicleId, int component, int command, int result, bool noReponseFromVehicle);
    void homePositionChanged(const QGeoCoordinate& currentHomePosition);
    void armedChanged(bool armed);
    void landedChanged();
    void flightModeChanged(const QString& flightMode);
    void flightModesChanged         (void);
    void coordinateChanged(const QGeoCoordinate& position);
    void homePositionReceivedChanged();
    //// Communication count
    ///
    void messagesReceivedChanged    ();
    void messagesSentChanged        ();
    void messagesLostChanged        ();
    /// Remote control RSSI changed  (0% - 100%)
    void remoteControlRSSIChanged(uint8_t rssi);

    void mavlinkRawImu(mavlink_message_t message);
    void mavlinkScaledImu1(mavlink_message_t message);
    void mavlinkScaledImu2(mavlink_message_t message);
    void mavlinkScaledImu3(mavlink_message_t message);
    void rollChanged();
    void pitchChanged();
    void headingChanged();
    void airSpeedChanged();
    void altitudeRelativeChanged();
    void engineSensor_1Changed();
    void engineSensor_2Changed();
    void postionChanged();
    void gpsChanged(bool valid);
    void linkChanged();
    void ekfChanged();
    void vibeChanged();
    void headingToHomeChanged();
    void distanceToHomeChanged();
    void currentWaypointChanged();
    void distanceToCurrentWaypointChanged();
    void batteryVoltageChanged();
    void batteryAmpeChanged();
    void groundSpeedChanged();
    void climbSpeedChanged();
    void altitudeAMSLChanged();
    void altitudeAGLChanged();
    void latGPSChanged();
    void lonGPSChanged();
    void hdopGPSChanged();
    void vdopGPSChanged();
    void courseOverGroundGPSChanged();
    void countGPSChanged();
    void lockGPSChanged();
    void uasChanged();
    void firmwareVersionChanged(void);
    void firmwareCustomVersionChanged(void);
    void vehicleUIDChanged();
    void messageSecurityChanged();
    /// Used internally to move sendMessage call to main thread
    void _sendMessageOnLinkOnThread(IOFlightController* link, mavlink_message_t message);    
    void textMessageReceived(int uasid, int componentid, int severity, QString text);
    void unhealthySensorsChanged(void);
    void sensorsPresentBitsChanged  (int sensorsPresentBits);
    void sensorsEnabledBitsChanged  (int sensorsEnabledBits);
    void sensorsHealthBitsChanged   (int sensorsHealthBits);
    void mavlinkStatusChanged();    
    void vehicleTypeChanged();
    void paramAirSpeedChanged();
    void paramLoiterRadiusChanged();
    void paramChanged(QString name);
    void rssiChanged();
    void pressABSChanged();
    void temperatureChanged();
    void propertiesModelChanged();
    void propertiesShowCountChanged();
    void paramsModelChanged();
public Q_SLOTS:
    void _loadDefaultParamsShow();
    void _setPropertyValue(QString name,QString value,QString unit);
    void _sendMessageOnLink(IOFlightController* link, mavlink_message_t message);
    void _mavlinkMessageReceived(mavlink_message_t message);
    void _sendGCSHeartbeat(void);
    void _checkCameraLink(void);
    void _sendGetParams(void);
    void _sendQGCTimeToVehicle(void);
    void requestDataStream(int messageID, int hz, int enable = 1);
    void _startPlanRequest(void);
    void _mavlinkMessageStatus(int uasId, uint64_t totalSent, uint64_t totalReceived, uint64_t totalLoss, float lossPercent);

public:
    IOFlightController* m_com = nullptr;
    FirmwarePlugin*      m_firmwarePlugin = nullptr;
    uint                _messagesReceived;
    uint                _messagesSent;
    uint                _messagesLost;
    uint8_t             _messageSeq;
    uint8_t             _compID;
    bool                _heardFrom;

    int _firmwareMajorVersion;
    int _firmwareMinorVersion;
    int _firmwarePatchVersion;
    int _firmwareCustomMajorVersion;
    int _firmwareCustomMinorVersion;
    int _firmwareCustomPatchVersion;
    FIRMWARE_VERSION_TYPE _firmwareVersionType;
    quint64 _uid;
private:
    Vehicle* m_uav = nullptr;
    UAS* m_uas;
    ParamsController *              m_paramsController;
    PlanController *                m_planController;
    FirmwarePluginManager*          m_firmwarePluginManager;
    // Queue Message
    QTimer                          _cameraLink;
    QTimer                          _mavParams;
    QTimer                          _mavHeartbeat;
    int                             _countHeartBeat = 0;
    QList<MavCommandQueueEntry_t>   _mavCommandQueue;
    QTimer                          _mavCommandAckTimer;
    int                             _mavCommandRetryCount;
    static const int                _mavCommandMaxRetryCount = 3;
    static const int                _mavCommandAckTimeoutMSecs = 3000;
    static const int                _mavCommandAckTimeoutMSecsHighLatency = 120000;
    void _handlePing(mavlink_message_t& message);
    void _handleHomePosition(mavlink_message_t& message);
    void _handleHeartbeat(mavlink_message_t& message);
    void _handleRadioStatus(mavlink_message_t& message);
    void _handleRCChannels(mavlink_message_t& message);
    void _handleRCChannelsRaw(mavlink_message_t& message);
    void _handleBatteryStatus(mavlink_message_t& message);
    void _handleBattery2Status(mavlink_message_t& message);
    void _handleSysStatus(mavlink_message_t& message);
    void _handleAutopilotVersion(mavlink_message_t& message);
    void _handleProtocolVersion(mavlink_message_t& message);
    void _handleRangeFinder(mavlink_message_t& message);
    void _handleWindCov(mavlink_message_t& message);
    void _handleVibration(mavlink_message_t& message);
    void _handleWind(mavlink_message_t& message);
    void _handleExtendedSysState(mavlink_message_t& message);
    void _handleCommandAck(mavlink_message_t& message);
    void _handleCommandLong(mavlink_message_t& message);
    void _handleHilActuatorControls(mavlink_message_t& message);
    void _handleGpsRawInt(mavlink_message_t& message);
    void _handleGlobalPositionInt(mavlink_message_t& message);
    void _handleAltitude(mavlink_message_t& message);
    void _handleVfrHud(mavlink_message_t& message);
    void _handleScaledPressure(mavlink_message_t& message);
    void _handleScaledPressure2(mavlink_message_t& message);
    void _handleScaledPressure3(mavlink_message_t& message);
    void _handleHighLatency2(mavlink_message_t& message);
    void _handleAttitudeWorker(double rollRadians, double pitchRadians, double yawRadians);
    void _handleAttitude(mavlink_message_t& message);
    void _handleAttitudeQuaternion(mavlink_message_t& message);
    void _handleAttitudeTarget(mavlink_message_t& message);
    void _handleDistanceSensor(mavlink_message_t& message);
    void _handleEstimatorStatus(mavlink_message_t& message);
    void _handleStatusText(mavlink_message_t& message, bool longVersion);
    void _handleOrbitExecutionStatus(const mavlink_message_t& message);
    void _handleEKFState(mavlink_message_t& message);
    void _handleRPMEngine(mavlink_message_t& message);
    void _handleRCIn(mavlink_message_t& message);
    void _handleServoOut(mavlink_message_t& message);
    void _handleNAVControllerOutput(mavlink_message_t& message);
    void _handleMavlinkLoggingData(mavlink_message_t& message);
    void _handleMavlinkLoggingDataAcked(mavlink_message_t& message);
    void _handleCameraImageCaptured(const mavlink_message_t& message);
    void _handleADSBVehicle(const mavlink_message_t& message);
    void _handlePMU(mavlink_message_t& message);
    void _updateArmed(bool armed);
    // data
    MAV_AUTOPILOT       _firmwareType = MAV_AUTOPILOT_ARDUPILOTMEGA;
    VEHICLE_MAV_TYPE    _vehicleType = MAV_TYPE_GENERIC;
    // data
    QString         _logFile;
    QGeoCoordinate  _coordinate;
    float           _paramAirSpeed = 0;
    float           _paramLoiterRadius = 0;
    float           _homeAltitude = 0;
    QGeoCoordinate  _homePosition;
    QString         _flightMode;
    int             _currentMessageCount;
    int             _messageCount;
    int             _currentErrorCount;
    int             _currentWarningCount;
    int             _currentNormalCount;
    MessageType_t   _currentMessageType;
    QString         _latestError;
    int             _updateCount;
    QString         _formatedMessage;
    int             _rcRSSI;
    double          _rcRSSIstore;
    bool            _autoDisconnect;    ///< true: Automatically disconnect vehicle when last connection goes away or lost heartbeat
    bool            _flying;
    bool            _landing;
    bool            _vtolInFwdFlight;
    uint32_t        _onboardControlSensorsPresent;
    uint32_t        _onboardControlSensorsEnabled;
    uint32_t        _onboardControlSensorsHealth;
    uint32_t        _onboardControlSensorsUnhealthy;
    bool            _gpsRawIntMessageAvailable = false;
    bool            _globalPositionIntMessageAvailable = false;
    double          _defaultCruiseSpeed;
    double          _defaultHoverSpeed;
    int             _telemetryRRSSI;
    int             _telemetryLRSSI;
    uint32_t        _telemetryRXErrors;
    uint32_t        _telemetryFixed;
    uint32_t        _telemetryTXBuffer;
    int             _telemetryLNoise;
    int             _telemetryRNoise;
    unsigned        _maxProtoVersion;
    bool            _vehicleCapabilitiesKnown;
    uint64_t        _capabilityBits;
    bool            _highLatencyLink = false;
    bool            _receivingAttitudeQuaternion = false;
    float           _roll = 0;
    float           _pitch = 0;
    float           _heading = 0;
    float           _airSpeed = 0;
    float           _climbRate = 0;
    float           _altitudeRelative = 0;
    float           _engineSensor_1 = 0;
    float           _engineSensor_2 = 0;
    bool            _link = false;
    int             _gpsFixedType = 0;
    QString         _ekf = "red";
    QString         _vibration = "red";
    float           _headingToHome = 0;
    float           _distanceToHome = 0;
    int             _currentWaypoint = 0;
    float           _distanceToCurrentWaypoint = 0;
    float           _batteryVoltage = 0;
    float           _batteryAmpe = 0;
    float           _groundSpeed = 0;
    float           _climbSpeed = 0;
    float           _altitudeAMSL = 0;
    float           _altitudeAGL = 0;

    float           _latGPS = 0;
    float           _lonGPS = 0;
    float           _hdopGPS = 0;
    float           _vdopGPS = 0;
    float           _courseOverGroundGPS = 0;
    int             _countGPS  = 0;
    QString         _lockGPS;
    QString         _messageSecurity = "MSG_INFO";

    int             _cameraLinkLast = 0;
    int             _linkHeartbeatRecv = 0;
    int             _id = 1;                    ///< Mavlink system id
    int             _defaultComponentId = MAV_COMP_ID_ALL;
    bool            _active;
    bool            _offlineEditingVehicle; ///< This Vehicle is a "disconnected" vehicle for ui use while offline editing

    bool    _armed = false;         ///< true: vehicle is armed
    uint8_t _base_mode = 0;     ///< base_mode from HEARTBEAT
    uint32_t _custom_mode = 0;  ///< custom_mode from HEARTBEAT
    bool _landed = true;

    uint64_t    _mavlinkSentCount       = 0;
    uint64_t    _mavlinkReceivedCount   = 0;
    uint64_t    _mavlinkLossCount       = 0;
    float       _mavlinkLossPercent     = 100.0f;
    float _pressABS = 0;
    int _temperature = 0;
    QList<Fact*> _propertiesModel;
    QList<Fact*> _paramsModel;
    QMap<QString,int> _paramsMap;
    int _propertiesShowCount = 0;
    bool _requestPlanAfterParams = false;
};

#endif // VEHICLE_H
void _missionLoadComplete(void);
