#include "PlateLogThread.h"

PlateLogThread::PlateLogThread(QObject *parent) : QObject(parent)
{
    m_mutexCapture = new QMutex();
    m_pauseCond = new QWaitCondition();
}
void PlateLogThread::setLogFile(QString logFile){
    m_logFile = logFile;
    printf("Start read file %s\r\n",m_logFile.toStdString().c_str());
    paused(false);
}
void PlateLogThread::appendLogFile(QString file,QString line){
    FileController::addLine(file.toStdString(),line.toStdString());
    Q_EMIT plateReaded(line);
}
void PlateLogThread::doWork(){
    while(!m_stop){
        m_mutexCapture->lock();
        if(m_pause)
        m_pauseCond->wait(m_mutexCapture); // in this place, your thread will stop to execute until someone calls resume
        m_mutexCapture->unlock();

        vector<string> lines = FileController::readFile(m_logFile.toStdString());
        printf("File contains %d lines\r\n",lines.size());
        for(int lineID=0; lineID< lines.size(); lineID++){
//            printf("Added line %s\r\n",lines[lineID].c_str());
            Q_EMIT plateReaded(QString::fromStdString(lines[lineID]));
        }
        paused(true);
    }
}
void PlateLogThread::paused(bool pause){
    if(pause == true){
        m_mutexCapture->lock();
        m_pause = true;
        m_mutexCapture->unlock();
    }else{
        m_mutexCapture->lock();
        m_pause = false;
        m_mutexCapture->unlock();
        m_pauseCond->wakeAll();
    }
}
