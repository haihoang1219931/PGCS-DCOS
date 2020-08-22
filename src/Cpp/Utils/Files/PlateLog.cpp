#include "PlateLog.h"

PlateLog::PlateLog(QObject *parent) : QObject(parent)
{
    m_readLog = new PlateLogThread(parent);
    m_threadReadLog = new QThread(parent);
    m_readLog->moveToThread(m_threadReadLog);
    connect(m_threadReadLog, SIGNAL(started()), m_readLog, SLOT(doWork()));
    connect(m_readLog, SIGNAL(plateReaded(QString)), this, SIGNAL(plateReaded(QString)));
}
void PlateLog::appendLogFile(QString file,QString line){
    m_readLog->appendLogFile(file,line);
}
void PlateLog::readLogFile(QString file){
    m_readLog->setLogFile(file);
    m_threadReadLog->start();
}
void PlateLog::pause(bool pause){
    m_readLog->paused(pause);
}
