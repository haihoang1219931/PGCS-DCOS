#include "VideoConverterThread.h"

VideoConverterThread::VideoConverterThread(QObject *parent) : QObject(parent)
{
    m_workerThread = new QThread(0);
    m_task = new VideoConverter(0);
    m_mutex = new QMutex();
    m_pauseCond = new QWaitCondition();
    m_task->moveToThread(m_workerThread);
    connect(m_workerThread, SIGNAL(started()), m_task, SLOT(doWork()));
    m_task->m_mutex = m_mutex;
    m_task->m_pauseCond = m_pauseCond;
}
VideoConverterThread::~VideoConverterThread()
{
    m_task->m_stop = true;
    m_workerThread->wait(100);
    m_workerThread->quit();
    if(!m_workerThread->wait(100)){
         m_workerThread->terminate();
        m_workerThread->wait(100);
    }
    printf("Copy thread stopped\r\n");
    m_task->deleteLater();
    m_workerThread->deleteLater();
    delete m_mutex;
    delete m_pauseCond;
}
void VideoConverterThread::start(){
    m_workerThread->start();
}
void VideoConverterThread::stop(){
    m_task->m_stop = true;
}
void VideoConverterThread::changeState(QString state){
    stateChanged(state);
}
