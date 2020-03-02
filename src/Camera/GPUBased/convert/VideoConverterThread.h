#ifndef VIDEOCONVERTERTHREAD_H
#define VIDEOCONVERTERTHREAD_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include "VideoConverter.h"
class VideoConverterThread : public QObject
{
    Q_OBJECT
public:
    explicit VideoConverterThread(QObject *parent = nullptr);
    virtual ~VideoConverterThread();
public:
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
Q_SIGNALS:
    void stateChanged(QString state);
public Q_SLOTS:
    void changeState(QString state);
public:
    QThread *m_workerThread = NULL;
    VideoConverter * m_task = NULL;
    QMutex *m_mutex = NULL;
    QWaitCondition *m_pauseCond = NULL;
};

#endif // VIDEOCONVERTERTHREAD_H
