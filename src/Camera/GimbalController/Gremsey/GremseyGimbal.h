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
    void setGimbalRate(float panRate,float tiltRate) override;
    void setEOZoom(QString command, int value) override;
    void snapShot() override;
    void changeTrackSize(float trackSize) override;
    void setDigitalStab(bool enable) override;
    void setLockMode(QString mode, QPoint location = QPoint(0,0)) override;
    void setRecord(bool enable) override;
    void setShare(bool enable) override;
Q_SIGNALS:

public Q_SLOTS:
    void handleAxisValueChanged(int axisID, float value);
    void handleButtonStateChanged(int buttonID, bool pressed);
    void sendQueryZoom();
    void handleQuery(QString data);
private:
    void enableDigitalZoom(bool enable);
private:
    QTimer* m_timerQueryZoom;
    GRGimbalController* m_gimbal;
    SensorController* m_sensor;
    QVector<int> m_zoom;
    QMap<int,int> m_mapZoom;
};

#endif // GREMSEYGIMBAL_H
