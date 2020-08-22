#include "MessageManager.h"

MessageManager::MessageManager(QObject *parent) : QObject(parent)
{

}
MessageManager::~MessageManager(){

}
void MessageManager::push(mavlink_message_t msg,QUEUE_ID queueID){
    if(queueID == QUEUE_ID::QUEUE_IN){
        while(m_queueIn.length() >= MAX_QUEUE_LENGTH){
            m_queueIn.removeFirst();
        }
        m_queueIn.push_back(msg);
    }else if(queueID == QUEUE_ID::QUEUE_OUT){
        while(m_queueOut.length() >= MAX_QUEUE_LENGTH){
            m_queueOut.removeFirst();
        }
        m_queueOut.push_back(msg);
    }
}
QList<mavlink_message_t> MessageManager::pop(QUEUE_ID queueID){
    QList<mavlink_message_t> ret;
    if(queueID == QUEUE_ID::QUEUE_OUT){
        if(m_queueOut.length() > 0){
            mavlink_message_t tmp = m_queueOut.front();
            m_queueOut.pop_front();
            ret.push_back(tmp);
        }
    }
    return ret;
}
