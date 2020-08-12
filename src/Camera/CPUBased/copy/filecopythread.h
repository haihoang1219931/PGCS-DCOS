#ifndef FILECOPYTHREAD_H
#define FILECOPYTHREAD_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include "FileCopy.h"
class FileCopyThread : public QObject
{
    Q_OBJECT
public:
    explicit FileCopyThread(QObject *parent = nullptr);
    virtual ~FileCopyThread();
public:
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
Q_SIGNALS:
    void processChanged(float copiedSize,float fileSize);
    void stateChanged(QString state);
public Q_SLOTS:
    void slotProcessChanged(float copiedSize,float fileSize);
    void slotChangeState(QString state);
    void setState(QString state);
    void copyFile(QString src,QString dst);
    void killQThread();
public:
    QThread *m_workerThread = NULL;
    FileCopy * m_task = NULL;
    QMutex *m_mutex = NULL;
    QWaitCondition *m_pauseCond = NULL;
};

#endif // FILECOPYTHREAD_H
