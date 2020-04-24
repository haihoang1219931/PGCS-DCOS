#ifndef GREMSEYGIMBAL_H
#define GREMSEYGIMBAL_H

#include <QObject>
#include <QTimer>
#include "../GimbalInterface.h"

class GRGimbalController;
class TCPClient;
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
    void setDigitalStab(bool enable) override;
    void setRecord(bool enable) override;
    void setShare(bool enable) override;
Q_SIGNALS:

public Q_SLOTS:
    void handleAxisValueChanged(int axisID, float value);
    void sendQueryZoom();
    void handleQueryZoom();
private:
    void enableDigitalZoom(bool enable);
private:
    QTimer* m_timerQueryZoom;
    GRGimbalController* m_gimbal;
    TCPClient* m_sensor;
};

#endif // GREMSEYGIMBAL_H
