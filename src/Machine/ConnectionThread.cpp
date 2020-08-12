#include "ConnectionThread.h"

ConnectionThread::ConnectionThread(QObject *parent) : QObject(parent)
{
    m_connection = new ConnectionChecking(parent);
    m_workerThread = new QThread(parent);
    m_connection->moveToThread(m_workerThread);
    connect(m_workerThread,SIGNAL(started()),m_connection,SLOT(doWork()));
    connect(m_connection,SIGNAL(stateChange(QString)),this,SLOT(changeState(QString)));
}
ConnectionThread::~ConnectionThread(){
    m_connection->m_stop = true;
    m_workerThread->quit();

    if(!m_workerThread->wait(1000)){
         m_workerThread->terminate();
        m_workerThread->wait(1000);
    }
    m_workerThread->deleteLater();
    m_connection->deleteLater();
}
void ConnectionThread::start(){
    m_connection->start();
    m_workerThread->start();
}
void ConnectionThread::stop(){
    m_workerThread->quit();
}
void ConnectionThread::changeState(QString state){
    stateChanged(state);
}
