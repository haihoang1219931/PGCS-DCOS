#ifndef GREMSEYGIMBAL_H
#define GREMSEYGIMBAL_H

#include <QObject>
#include <QTimer>
#include "../GimbalInterface.h"
#include <QMap>
#include <QVector>
class GRGimbalController;
class SensorController;
class GremseyGimbal : public GimbalInterface
{
    Q_OBJECT
public:
    explicit GremseyGimbal(GimbalInterface *parent = nullptr);
public:
    void setJoystick(JoystickThreaded* joystick) override;
    void connectToGimbal(Config* config = nullptr) override;
    void disconnectGimbal() override;
    void changeSensor(QString sensorID) override;
    void setSensorColor(QString sensorID,QString colorMode) override;
    void setGimbalRate(float panRate,float tiltRate) override;
    void setEOZoom(QString command, float value = 0) override;
    void setIRZoom(QString command) override;
    void snapShot() override;
    void changeTrackSize(float trackSize) override;
    void setDigitalStab(bool enable) override;
    void setLockMode(QString mode, QPointF location = QPointF(0,0)) override;
    void setRecord(bool enable) override;
    void setShare(bool enable) override;
    void setGimbalPreset(QString mode) override;
    void setGimbalMode(QString mode) override;
    void setGimbalPos(float panPos,float tiltPos) override;

    Q_INVOKABLE void setVehicle(Vehicle* vehicle) override;
Q_SIGNALS:

public Q_SLOTS:
    void handleAxisValueChanged(int axisID, float value);
    void handleButtonStateChanged(int buttonID, bool pressed);
    void sendQueryZoom();
    void handleQuery(QString data);
    void handleGimbalMessage(mavlink_message_t message);
    void handleVehicleMessage(mavlink_message_t message);
    void handleVehicleLinkChanged();
    void handleGimbalModeChanged(QString mode) override;
    void handleGimbalSetModeFail() override;

private:
    void enableDigitalZoom(bool enable);
private:
    QTimer* m_timerQueryZoom;
    GRGimbalController* m_gimbal;
    SensorController* m_sensor;
    QVector<int> m_zoom;
    QMap<int,int> m_mapZoom;
    Vehicle* m_vehicle = nullptr;
    float m_presetAngleSet_Pan = 0;
    float m_presetAngleSet_Tilt = 0;
//    QString m_modePreset = "OFF";
    QString m_gimbalCurrentMode = "UNKNOWN";
    float m_panRateJoystick = 0;
    float m_tiltRateJoystick = 0;
};

#endif // GREMSEYGIMBAL_H
