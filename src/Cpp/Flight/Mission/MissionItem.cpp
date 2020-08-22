#include "MissionItem.h"

MissionItem::MissionItem(QObject *parent) : QObject(parent)
{

}
MissionItem::MissionItem(int             sequenceNumber,
                         MAV_CMD         command,
                         MAV_FRAME       frame,
                         float          param1,
                         float          param2,
                         float          param3,
                         float          param4,
                         float          param5,
                         float          param6,
                         float          param7,
                         bool            autoContinue,
                         bool            isCurrentItem,
                         QObject*        parent){

    // Need a good command and frame before we start passing signals around
    m_command = command;
    m_frame = frame;
    m_param1 = param1;
    m_param2 = param2;
    m_param3 = param3;
    m_param4 = param4;
    m_param5 = param5;
    m_param6 = param6;
    m_param7 = param7;
    m_autoContinue = autoContinue;
    m_isCurrentItem = isCurrentItem;
    m_seq = sequenceNumber;
}
