#ifndef CONNECTIONCHECKING_H
#define CONNECTIONCHECKING_H

#include <QObject>
#include <QTime>
#include <QProcess>
#include <iostream>
#include <vector>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#ifdef __linux__
    //linux code goes here
    #include <sys/time.h>
#elif _WIN32
    // windows code goes here
#else

#endif

using namespace std;
class ConnectionChecking : public QObject
{
    Q_OBJECT

public:
    explicit ConnectionChecking(QObject *parent = nullptr);
    virtual ~ConnectionChecking();

    void msleep(int ms);
public:
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
Q_SIGNALS:
    void stateChange(QString state);
public Q_SLOTS:
    void doWork();
public:
    int count = 0;
    string m_address;
    bool m_stop = true;
};

#endif // CONNECTIONCHECKING_H
