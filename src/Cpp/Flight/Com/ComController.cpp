#include "ComController.h"

ComController::ComController(QObject *parent) : QObject(parent)
{
    printf("Init ComController\r\n");
    m_comTCP = new ComTCP();
    m_comTCPThread = new QThread(0);
    m_comTCP->moveToThread(m_comTCPThread);
    connect(m_comTCPThread, SIGNAL(started()), m_comTCP, SLOT(doWork()));
}

ComController::~ComController()
{
    m_comTCP->m_stop = true;
    m_comTCPThread->wait(100);
    m_comTCPThread->quit();
    if(!m_comTCPThread->wait(100)){
         m_comTCPThread->terminate();
        m_comTCPThread->wait(100);
    }
    printf("ComTCP thread stopped\r\n");
    m_comTCPThread->deleteLater();
    m_comTCP->deleteLater();
}
void ComController::setAddress(QString ip, int port){
    m_comTCP->setAddress(ip,port);
}
void ComController::start(){
    m_comTCPThread->start();
}
void ComController::stop(){
    m_comTCP->m_stop = true;
}
