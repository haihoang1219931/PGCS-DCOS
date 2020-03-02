#ifndef NETWORKPARAMETERSCONTEXT_H
#define NETWORKPARAMETERSCONTEXT_H

#include <QObject>
#include <QHostAddress>
#include <stdio.h>
#include <iostream>
using namespace std;

class NetworkInterfaceParameters: public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool isStatic READ isStatic)
    Q_PROPERTY(bool hasChanged READ hasChanged)
    Q_PROPERTY(string ipAddress READ ipAddress)
    Q_PROPERTY(string gateWay READ gateWay)
    Q_PROPERTY(string subnet READ subnet)
    Q_PROPERTY(int sendPort READ sendPort)
    Q_PROPERTY(int listenPort READ listenPort)
public:
    bool m_isStatic;
    bool m_hasChanged;
    QHostAddress m_ipAddress;
    QHostAddress m_gateWay;
    QHostAddress m_subnet;
    int m_sendPort;
    int m_listenPort;
    bool isStatic(){
        return m_isStatic;
    }
    bool hasChanged(){
        return m_hasChanged;
    }
    string ipAddress(){
        return m_ipAddress.toString().toStdString();
    }
    string gateWay(){
        return m_gateWay.toString().toStdString();
    }
    string subnet(){
        return m_subnet.toString().toStdString();
    }
    int sendPort(){
        return m_sendPort;
    }
    int listenPort(){
        return m_listenPort;
    }
    explicit NetworkInterfaceParameters(QObject *parent = 0);
    virtual ~NetworkInterfaceParameters();
Q_SIGNALS:
    void NotifyPropertyChanged(QString name);
};
class NetworkParametersContext: public QObject
{
    Q_OBJECT
    Q_PROPERTY(NetworkInterfaceParameters *eth0 READ eth0)
    Q_PROPERTY(NetworkInterfaceParameters *eth1 READ eth1)
public:
    NetworkInterfaceParameters* m_eth0;
    NetworkInterfaceParameters* m_eth1;
    NetworkInterfaceParameters* eth0(){
        return m_eth0;
    }
    NetworkInterfaceParameters* eth1(){
        return m_eth1;
    }
    explicit NetworkParametersContext(QObject *parent = 0);
    virtual ~NetworkParametersContext();
};
#endif // NETWORKPARAMETERSCONTEXT_H
