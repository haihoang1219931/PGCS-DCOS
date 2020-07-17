#ifndef QUADPLANEFIRMWARE_H
#define QUADPLANEFIRMWARE_H

#include <QObject>
#include <QVector>
#include <QStringList>
#include <QMap>
#include "../FirmwarePlugin.h"
class Vehicle;
class QuadPlaneFirmware : public FirmwarePlugin
{
    Q_OBJECT
public:
    enum Mode {
        MANUAL        = 0,
        CIRCLE        = 1,
        STABILIZE     = 2,
        TRAINING      = 3,
        ACRO          = 4,
        FLY_BY_WIRE_A = 5,
        FLY_BY_WIRE_B = 6,
        CRUISE        = 7,
        AUTOTUNE      = 8,
        RESERVED_9    = 9,  // RESERVED FOR FUTURE USE
        AUTO          = 10,
        RTL           = 11,
        LOITER        = 12,
        RESERVED_13   = 13, // RESERVED FOR FUTURE USE
        RESERVED_14   = 14, // RESERVED FOR FUTURE USE
        GUIDED        = 15,
        INITIALIZING  = 16,
        QSTABILIZE    = 17,
        QHOVER        = 18,
        QLOITER       = 19,
        QLAND         = 20,
        QRTL          = 21,
        modeCount
    };
    QuadPlaneFirmware(Vehicle* vehicle = nullptr);
    QString flightMode(int flightModeId) override;
    bool flightModeID(QString flightMode,int* base_mode,int* custom_mode) override;
    void initializeVehicle() override;
    /// Returns the flight mode which the vehicle will be in if it is performing a goto location
    QString gotoFlightMode() const override;
    /// Sets base_mode and custom_mode to specified flight mode.
    ///     @param[out] base_mode Base mode for SET_MODE mavlink message
    ///     @param[out] custom_mode Custom mode for SET_MODE mavlink message
    bool setFlightMode(const QString& flightMode, uint8_t* base_mode, uint32_t* custom_mode) override;

    /// Command vehicle to return to launch
    void commandRTL() override;

    /// Command vehicle to land at current location
    void commandLand() override;

    /// Command vehicle to takeoff from current location
    void commandTakeoff(double altitudeRelative) override;

    /// @return The minimum takeoff altitude (relative) for guided takeoff.
    double minimumTakeoffAltitude() override;

    /// Command vehicle to move to specified location (altitude is included and relative)
    void commandGotoLocation(const QGeoCoordinate& gotoCoord) override;

    /// Command vehicle to change altitude
    ///     @param altitudeChange If > 0, go up by amount specified, if < 0, go down by amount specified
    void commandSetAltitude(double newAltitude) override;
    /// Command vehicle to change speed
    ///     @param speedChange If > 0, go up by amount specified, if < 0, go down by amount specified
    void commandChangeSpeed(double speedChange) override;
    /// Command vehicle to orbit given center point
    ///     @param centerCoord Orit around this point
    ///     @param radius Distance from vehicle to centerCoord
    ///     @param amslAltitude Desired vehicle altitude
    void commandOrbit(const QGeoCoordinate& centerCoord, double radius, double amslAltitude) override;

    /// Command vehicle to pause at current location. If vehicle supports guide mode, vehicle will be left
    /// in guided mode after pause.
    void pauseVehicle() override;

    /// Command vehicle to kill all motors no matter what state
    void emergencyStop() override;

    /// Command vehicle to abort landing
    void abortLanding(double climbOutAltitude) override;

    void startMission() override;

    /// command start engine
    void startEngine() override;

    /// Alter the current mission item on the vehicle
    void setCurrentMissionSequence( int seq) override;

    /// Reboot vehicle
    void rebootVehicle() override;

    /// Clear Messages
    void clearMessages() override;

    void triggerCamera() override;
    void sendPlan(QString planFile) override;

    /// Used to check if running current version is equal or higher than the one being compared.
    //  returns 1 if current > compare, 0 if current == compare, -1 if current < compare
    int versionCompare(QString& compare) override;
    int versionCompare(int major, int minor, int patch) override;

    /// Test motor
    ///     @param motor Motor number, 1-based
    ///     @param percent 0-no power, 100-full power
    void motorTest(int motor, int percent) override;

    /// Set home postion
    ///     @param lat: latitude
    ///     @param lon: longitude
    ///     @param alt: altitude
    void setHomeHere(float lat, float lon, float alt) override;

    void setGimbalMode(QString mode) override;

Q_SIGNALS:

public Q_SLOTS:
    void sendJoystickData();
    void sendClearRC();
    void handleJSButton(int id, bool clicked) override;
    void handleUseJoystick(bool enable) override;
    void handleJoystickConnected(bool connected);
private:
    float convertRC(float input, int channel);
};

#endif // QUADPLANEFIRMWARE_H
