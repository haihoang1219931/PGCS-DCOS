#include "MissionController.h"

MissionController::MissionController(QObject *parent) : QObject(parent)
{

}
Vehicle* MissionController::vehicle(){
    return m_vehicle;
}
void MissionController::setVehicle(Vehicle* vehicle){
    m_vehicle = vehicle;
    _connectToMavlink();
}
//void PlanController::setMissionItems(QQmlListProperty<MissionItem*> missionItems){

//}
void MissionController::_connectToMavlink(void)
{
    connect(m_vehicle, SIGNAL(mavlinkMessageReceived(mavlink_message_t)), this, SLOT(_mavlinkMessageReceived(mavlink_message_t)));
}

void MissionController::_disconnectFromMavlink(void)
{
    disconnect(m_vehicle, SIGNAL(mavlinkMessageReceived(const mavlink_message_t&)), this, SLOT(_mavlinkMessageReceived(const mavlink_message_t&)));
}
void MissionController::_mavlinkMessageReceived(const mavlink_message_t& message)
{
    switch (message.msgid) {
    case MAVLINK_MSG_ID_MISSION_CURRENT:
        _handleMissionCurrent(message);
        break;

    case MAVLINK_MSG_ID_HEARTBEAT:
        _handleHeartbeat(message);
        break;
    }
}

void MissionController::_handleMissionCurrent(const mavlink_message_t& message)
{
    mavlink_mission_current_t missionCurrent;

    mavlink_msg_mission_current_decode(&message, &missionCurrent);
    if (missionCurrent.seq != _currentMissionIndex) {
//        printf("MissionController._handleMissionCurrent[%d]\r\n",missionCurrent.seq);
        _currentMissionIndex = missionCurrent.seq;
       currentIndexChanged(_currentMissionIndex);
       m_vehicle->setCurrentWaypoint(_currentMissionIndex);
    }else{
        if(m_vehicle->flightMode() == "RTL"){
//            printf("MissionController._handleMissionCurrent[%d] at RTL\r\n",1);
           currentIndexChanged(0);
           m_vehicle->setCurrentWaypoint(0);
        }else{
//            printf("MissionController._handleMissionCurrent[%d] at %s\r\n",
//                   _currentMissionIndex,
//                   m_vehicle->flightMode().toStdString().c_str());
           currentIndexChanged(_currentMissionIndex);
           m_vehicle->setCurrentWaypoint(_currentMissionIndex);
        }
    }
}

void MissionController::_handleHeartbeat(const mavlink_message_t& message)
{
    Q_UNUSED(message);
}
