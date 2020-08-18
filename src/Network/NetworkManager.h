#ifndef NETWORKMANAGER_H
#define NETWORKMANAGER_H

#include <QObject>
#include <QNetworkConfigurationManager>
#include <QNetworkAccessManager>
#include <QNetworkSession>
#include <QList>
#include <QQmlListProperty>
#include <QtCore/QUuid>
#include <QtDBus/QDBusArgument>
#include <QtDBus/QDBusConnection>
#include <QtDBus/QDBusInterface>
#include <QtDBus/QDBusMetaType>
#include <QtDBus/QDBusReply>
#include <QtCore/QDebug>
#include <QtEndian>
#include <QTimer>
#include "NetworkInfo.h"

class NetworkInterface: public NetworkInfo{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<NetworkInfo> listNetwork READ listNetwork NOTIFY listNetworkChanged)    
public:
    explicit NetworkInterface(QObject *parent = nullptr){}
    QQmlListProperty<NetworkInfo> listNetwork()
    {
        return QQmlListProperty<NetworkInfo>(this, m_listNetwork);
    }
    QList<NetworkInfo *> getListNetwork(){
        return m_listNetwork;
    }
    void append(NetworkInfo* net){
        m_listNetwork.append(net);
        Q_EMIT listNetworkChanged();
    }

Q_SIGNALS:
    void listNetworkChanged();    
private:
    QList<NetworkInfo *> m_listNetwork;
};

class NetworkManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<NetworkInterface> listAccess READ listAccess NOTIFY listAccessChanged)
    Q_PROPERTY(QQmlListProperty<NetworkInterface> listSetting READ listSetting NOTIFY listSettingChanged)
public:
    explicit NetworkManager(QObject *parent = nullptr);
    QQmlListProperty<NetworkInterface> listAccess()
    {
        return QQmlListProperty<NetworkInterface>(this, m_listAccess);
    }
    QQmlListProperty<NetworkInterface> listSetting()
    {
        return QQmlListProperty<NetworkInterface>(this, m_listSetting);
    }

    static void expose();
    void setPass(QString pass){
        m_pass = pass;
    }
    void getAccessPointInfo(NetworkInfo* accessPoint);
    void getListAccess(NetworkInterface *interfaceName);
    void getListSetting(NetworkInterface *interfaceName);
    void getConnectionSetting(NetworkInterface *interface,QString settingPath, NetworkInfo *connection);
    bool connectSetting(NetworkInfo* setting, bool connect);
    bool connectAcessPoint(NetworkInfo* accessPoint, bool connect);
    QString convertIP(QString type,uint ipv4);
public:
    Q_INVOKABLE void reloadListAccess();
    Q_INVOKABLE void reloadListSetting();
    Q_INVOKABLE void connectNetwork(QString bearerTypeName, QString name,bool connect);
    Q_INVOKABLE void insertWLANPass(QString pass);
    Q_INVOKABLE void deleteSetting(QString setting);
    Q_INVOKABLE QVariant getConnectionSetting(QString settingPath);
Q_SIGNALS:
    void listAccessChanged();
    void listSettingChanged();
    void needWLANPass();
public Q_SLOTS:
    void reloadAddresses();
private:
    NetworkInfo* m_currentAccessPoint = nullptr;
    QNetworkConfigurationManager m_mgr;
    QNetworkSession* m_session = nullptr;
    QList<NetworkInterface *> m_listAccess;
    QList<NetworkInterface *> m_listSetting;
    QString m_pass = "1";
    QTimer m_timerReloadAddress;
};

#endif // NETWORKMANAGER_H
