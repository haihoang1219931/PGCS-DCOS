#ifndef COMPUTER_H
#define COMPUTER_H

#include <QObject>
#include <QProcess>
#include <QTimer>
#include <QApplication>
#include <QDir>
#include <QFileSystemWatcher>
#include <QFileInfoList>
#include <QStorageInfo>
#include <QDebug>
#include <iostream>
#include "../Utils/filenameutils.h"
#ifdef __linux__
    //linux code goes here
    #include <sys/statvfs.h>
    #include <sys/types.h>
    #include <sys/sysinfo.h>

#elif _WIN32
    // windows code goes here
#else

#endif
#include <stdio.h>
#define DEBUG_OFF
using namespace std;
class COMPUTER_INFO: public QObject{
    Q_OBJECT
public:
    COMPUTER_INFO(QObject*parent = 0);
    virtual ~COMPUTER_INFO();
    Q_INVOKABLE QString media();
    Q_INVOKABLE QString user();
    Q_INVOKABLE QString freeSpace();
    Q_INVOKABLE QString version();
    Q_INVOKABLE QString homeFolder();
    Q_INVOKABLE void calibrateJoystick();
    Q_INVOKABLE void setSystemTime();
    Q_INVOKABLE void restartApplication();
    Q_INVOKABLE void quitApplication();
    Q_INVOKABLE void restartComputer();
    Q_INVOKABLE void shutdownComputer();
    Q_INVOKABLE void openFolder(QString folder);
    Q_INVOKABLE int fileSize(QString fileName);
    Q_INVOKABLE void fileCopy(QString fileSrc,QString fileDst);
    Q_INVOKABLE void fileDelete(QString fileName);

Q_SIGNALS:
    void usbAdded(QStringList mediaFolder);
    void usbRemoved(QStringList mediaFolder);
public Q_SLOTS:
    void handleUsbDetected(QString mediaFolder);
    void checkSystemMem();
public:
    QFileSystemWatcher* m_usbWatcher;
    QStringList m_listUSB;
    QTimer* m_timer;
    int m_ramTotal = -1;
    int m_ramUsed = -1;
};
#endif // COMPUTER_H
