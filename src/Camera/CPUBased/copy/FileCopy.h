#ifndef FILECOPY_H
#define FILECOPY_H

#include <QObject>
#include <QMutex>
#include <QWaitCondition>
#include <QFile>
#include <QFileInfo>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
//#include <unistd.h>
class FileCopy: public QObject
{
    Q_OBJECT
public:
    explicit FileCopy(QObject *parent = nullptr);
    virtual ~FileCopy();
public:
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
    Q_INVOKABLE void setState(QString state);
    void copyFile(QString src,QString dst);
    void msleep(int ms);
Q_SIGNALS:
    void stopped();
    void stateChange(QString state);
    void processChanged(float copiedSize,float fileSize);
public Q_SLOTS:
    void doWork();
public:
    QString m_src;
    QString m_dst;
    QMutex *m_mutex;
    QWaitCondition *m_pauseCond;
    bool m_pause = false;
    bool m_stop = false;
};

#endif // FILECOPY_H
