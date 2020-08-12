#include "filecopythread.h"

FileCopyThread::FileCopyThread(QObject *parent) : QObject(parent)
{
    m_workerThread = new QThread(0);
    m_task = new FileCopy(0);
    m_mutex = new QMutex();
    m_pauseCond = new QWaitCondition();
    m_task->moveToThread(m_workerThread);
    connect(m_workerThread, SIGNAL(started()), m_task, SLOT(doWork()));
    connect(m_task, SIGNAL(processChanged(float,float)), this, SLOT(slotProcessChanged(float,float)));
    connect(m_task, SIGNAL(stateChange(QString)), this, SLOT(slotChangeState(QString)));
    connect(m_task, SIGNAL(stopped()), this, SLOT(killQThread()));
    m_task->m_mutex = m_mutex;
    m_task->m_pauseCond = m_pauseCond;
}
FileCopyThread::~FileCopyThread()
{
    printf("Destroying Copy thread\r\n");
    m_task->deleteLater();
    m_workerThread->deleteLater();
    delete m_mutex;
    delete m_pauseCond;
}
void FileCopyThread::start(){
    m_workerThread->start();
}
void FileCopyThread::stop(){
    m_task->m_stop = true;
}
void FileCopyThread::setState(QString state){
    m_task->setState(state);
}
void FileCopyThread::slotChangeState(QString state){
    stateChanged(state);
}
void FileCopyThread::slotProcessChanged(float copiedSize,float fileSize){
    processChanged(copiedSize,fileSize);
}
void FileCopyThread::copyFile(QString src,QString dst){
    m_task->copyFile(src,dst);
}
void FileCopyThread::killQThread(){
    if(!m_workerThread->wait(100)){
         m_workerThread->terminate();
        m_workerThread->wait();
    }
    printf("Copy thread stopped\r\n");
    stateChanged("COPY_DONE");
}
