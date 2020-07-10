#ifndef FIRMWAREPLUGIN_H
#define FIRMWAREPLUGIN_H

#include <QObject>
#include <QVector>
#include <QStringList>
#include <QMap>
#include <QGeoCoordinate>
#include <QTimer>

class Vehicle;
class Fact;
class FirmwarePlugin : public QObject
{
    Q_OBJECT
public:
    FirmwarePlugin(Vehicle* vehicle = nullptr);
    typedef enum {
        SetFlightModeCapability =           1 << 0, ///< FirmwarePlugin::setFlightMode method is supported
        PauseVehicleCapability =            1 << 1, ///< Vehicle supports pausing at current location
        commandCapability =              1 << 2, ///< Vehicle supports guided mode commands
        OrbitModeCapability =               1 << 3, ///< Vehicle supports orbit mode
        TakeoffVehicleCapability =          1 << 4, ///< Vehicle supports guided takeoff
    } FirmwareCapabilities;
    virtual void setVehicle(Vehicle *vehicle);
    virtual void loadFromFile(QString fileName);
    virtual void saveToFile(QString fileName,QList<Fact*> _listParamShow);
    virtual QList<Fact*> listParamsShow();
    /// Returns the list of available flight modes. Flight modes can be different in normal/advanced ui mode.
    /// Call will be made again if advanced mode changes.
    virtual QString rtlAltParamName();
    virtual QString airSpeedParamName();
    virtual QString loiterRadiusParamName();
    virtual QString flightMode(int flightModeId);
    virtual bool flightModeID(QString flightMode,int* base_mode,int* custom_mode);
    /// Called when Vehicle is first created to perform any firmware specific setup.
    virtual void initializeVehicle(Vehicle* vehicle);
    /// Returns the list of available flight modes. Flight modes can be different in normal/advanced ui mode.
    /// Call will be made again if advanced mode changes.
    virtual QStringList flightModes();
    virtual QStringList flightModesOnAir();
    /// Used to determine whether a vehicle has a gimbal.
    ///     @param[out] rollSupported Gimbal supports roll
    ///     @param[out] pitchSupported Gimbal supports pitch
    ///     @param[out] yawSupported Gimbal supports yaw
    /// @return true: vehicle has gimbal, false: gimbal support unknown
    virtual bool hasGimbal(Vehicle* vehicle, bool& rollSupported, bool& pitchSupported, bool& yawSupported);

    /// Returns true if the vehicle is a VTOL
    virtual bool isVtol(const Vehicle* vehicle) const;

    virtual void sendHomePosition(Vehicle* vehicle,QGeoCoordinate location);
    /// Sets base_mode and custom_mode to specified flight mode.
    ///     @param[out] base_mode Base mode for SET_MODE mavlink message
    ///     @param[out] custom_mode Custom mode for SET_MODE mavlink message
    virtual bool setFlightMode(const QString& flightMode, uint8_t* base_mode, uint32_t* custom_mode);

    /// Returns The flight mode which indicates the vehicle is paused
    virtual QString pauseFlightMode(void) const { return QString(); }

    /// Returns the flight mode for running missions
    virtual QString missionFlightMode(void) const { return QString(); }

    /// Returns the flight mode for RTL
    virtual QString rtlFlightMode(void) const { return QString(); }

    /// Returns the flight mode for Land
    virtual QString landFlightMode(void) const { return QString(); }

    /// Returns the flight mode to use when the operator wants to take back control from autonomouse flight.
    virtual QString takeControlFlightMode(void) const { return QString(); }

    /// Returns whether the vehicle is in guided mode or not.
    virtual bool iscommand(const Vehicle* vehicle) const;

    /// Returns the flight mode which the vehicle will be in if it is performing a goto location
    virtual QString gotoFlightMode(void) const;

    /// Set guided flight mode
    virtual void setcommand(Vehicle* vehicle, bool command);

    /// Causes the vehicle to stop at current position. If guide mode is supported, vehicle will be let in guide mode.
    /// If not, vehicle will be left in Loiter.
    virtual void pauseVehicle(Vehicle* vehicle);

    /// Command vehicle to return to launch
    virtual void commandRTL(void);

    /// Command vehicle to land at current location
    virtual void commandLand(void);

    /// Command vehicle to takeoff from current location
    virtual void commandTakeoff(Vehicle* vehicle, double altitudeRelative);

    /// @return The minimum takeoff altitude (relative) for guided takeoff.
    virtual double minimumTakeoffAltitude(void);

    /// Command vehicle to move to specified location (altitude is included and relative)
    virtual void commandGotoLocation(Vehicle *vehicle,const QGeoCoordinate& gotoCoord);

    /// Command vehicle to change altitude
    ///     @param altitudeChange If > 0, go up by amount specified, if < 0, go down by amount specified
    virtual void commandChangeAltitude(double altitudeChange);
    /// Command vehicle to change altitude
    ///     @param altitudeChange If > 0, go up by amount specified, if < 0, go down by amount specified
    virtual void commandSetAltitude(Vehicle *vehicle,double newAltitude);
    /// Command vehicle to change speed
    ///     @param speedChange If > 0, go up by amount specified, if < 0, go down by amount specified
    virtual void commandChangeSpeed(Vehicle* vehicle,double speedChange);
    /// Command vehicle to orbit given center point
    ///     @param centerCoord Orit around this point
    ///     @param radius Distance from vehicle to centerCoord
    ///     @param amslAltitude Desired vehicle altitude
    virtual void commandOrbit(const QGeoCoordinate& centerCoord, double radius, double amslAltitude);

    /// Command vehicle to pause at current location. If vehicle supports guide mode, vehicle will be left
    /// in guided mode after pause.
    virtual void pauseVehicle(void);

    /// Command vehicle to kill all motors no matter what state
    virtual void emergencyStop(void);

    /// Command vehicle to abort landing
    virtual void abortLanding(double climbOutAltitude);

    virtual void startMission(Vehicle* vehicle);

    virtual void startEngine(Vehicle* vehicle);

    /// Alter the current mission item on the vehicle
    virtual void setCurrentMissionSequence(Vehicle* vehicle, int seq);

    /// Reboot vehicle
    virtual void rebootVehicle();

    /// Clear Messages
    virtual void clearMessages();

    virtual void triggerCamera(void);
    virtual void sendPlan(QString planFile);

    /// Used to check if running current version is equal or higher than the one being compared.
    //  returns 1 if current > compare, 0 if current == compare, -1 if current < compare
    virtual int versionCompare(QString& compare);
    virtual int versionCompare(int major, int minor, int patch);

    /// Test motor
    ///     @param motor Motor number, 1-based
    ///     @param percent 0-no power, 100-full power
    virtual void motorTest(Vehicle* vehicle,int motor, int percent);

    /// Set home postion
    ///     @param lat: latitude
    ///     @param lon: longitude
    ///     @param alt: altitude
    virtual void setHomeHere(Vehicle* vehicle,float lat, float lon, float alt);

Q_SIGNALS:

public Q_SLOTS:
    virtual void handleJSButton(int id, bool clicked){}
    virtual void handleUseJoystick(bool enable){}
public:
    QString m_rtlAltParamName;
    QString m_airSpeedParamName;
    QString m_loiterRadiusParamName;
    QMap<int,QString> m_mapFlightMode;
    QMap<int,QString> m_mapFlightModeOnAir;
    QMap<int,QString> m_mapFlightModeOnGround;
    QList<Fact*> _listParamShow;
    Vehicle *m_vehicle = nullptr;
    QTimer m_joystickTimer;
    QTimer m_joystickClearRCTimer;
    int m_sendClearRCCount = 0;
    int m_sendClearRCMax = 5;
};

#endif // FIRMWAREPLUGIN_H
