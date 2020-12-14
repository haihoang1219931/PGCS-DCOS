#ifndef MISSIONCONTROLLER_H
#define MISSIONCONTROLLER_H

#include <QObject>
#include "../Vehicle/Vehicle.h"
class MissionController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool forceCurrentWP READ forceCurrentWP WRITE setForceShowCurrentWP)
public:
    MissionController(Vehicle *vehicle);
    void _handleMissionCurrent(const mavlink_message_t& message);
    void _handleHeartbeat(const mavlink_message_t& message);
    void setForceShowCurrentWP(bool force);
protected:
    void _connectToMavlink(void);
    void _disconnectFromMavlink(void);
Q_SIGNALS:
    void currentIndexChanged(int sequence);
public Q_SLOTS:
    void _mavlinkMessageReceived(const mavlink_message_t& message);
public:
    Vehicle* m_vehicle;
private:
    int _currentMissionIndex = -1;
    int _lastCurrentIndex = -1;
    bool m_forceShowCurrentWP = false;

    bool forceCurrentWP();
};

#endif // MISSIONCONTROLLER_H
