#ifndef PLATELOGTHREAD_H
#define PLATELOGTHREAD_H

#include <QObject>
#include <QMutex>
#include <QWaitCondition>
#include <QMap>
#include "FileControler.h"
class PlateLogThread : public QObject
{
    Q_OBJECT
public:
    explicit PlateLogThread(QObject *parent = nullptr);
    void setLogFile(QString logFile);
    void appendLogFile(QString file,QString line);
Q_SIGNALS:
    void plateReaded(QString logLine);
    void readDone();
public Q_SLOTS:
    void doWork();
    void paused(bool pause);
private:
    QMutex* m_mutexCapture;
    QWaitCondition *m_pauseCond;
    QString m_logFile;
    bool m_stop = false;
    bool m_pause = false;
};

#endif // PLATELOGTHREAD_H
