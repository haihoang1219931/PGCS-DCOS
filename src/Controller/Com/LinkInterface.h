#ifndef LINKINTERFACE_H
#define LINKINTERFACE_H

#include <QObject>
#include <QJsonDocument>
#include <QJsonObject>
#include <vector>
#include <iostream>
#include "../../Setting/config.h"
using namespace std;
class LinkInterface : public QObject
{
    Q_OBJECT
public:
    explicit LinkInterface(QObject *parent = nullptr);
    virtual bool isOpen(){return false;}
    virtual void loadConfig(Config* config){Q_UNUSED(config)}
    virtual bool getStatus() {return status;}
    virtual void sendData(vector<unsigned char> msg){Q_UNUSED(msg);}
    virtual void writeBytesSafe(const char *bytes, int length){Q_UNUSED(bytes);Q_UNUSED(length);}
Q_SIGNALS:
    void statusChanged(bool);
    void hasReadSome(QByteArray msg);
public Q_SLOTS:
    virtual void closeConnection(){}
    virtual void connect2host(){}
    virtual void readyRead(){}
    virtual void connected(){}
    virtual void disconnected(){}
    virtual void connectionTimeout(){}
public:
    QString name;
    QString type;
    bool status;
};

#endif // LINKINTERFACE_H
