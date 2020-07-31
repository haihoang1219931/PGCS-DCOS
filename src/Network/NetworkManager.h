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
#include "NetworkInfo.h"
typedef QMap<QString, QMap<QString, QVariant> > Connection;
Q_DECLARE_METATYPE(Connection)
Q_DECLARE_METATYPE(QList<uint>);
Q_DECLARE_METATYPE(QList<QList<uint> >);
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
    Q_PROPERTY(QQmlListProperty<NetworkInterface> listInterface READ listInterface NOTIFY listInterfaceChanged)
public:
    explicit NetworkManager(QObject *parent = nullptr);
    QQmlListProperty<NetworkInterface> listInterface()
    {
        return QQmlListProperty<NetworkInterface>(this, m_listInterface);
    }
    Q_INVOKABLE void reload();
    static void expose();
    void setPass(QString pass){
        m_pass = pass;
    }
    QString getConnection(QString settingPath, Connection *found_connection);
public:
    Q_INVOKABLE void connectNetwork(QString name,bool connect);
Q_SIGNALS:
    void listInterfaceChanged();
public Q_SLOTS:
private:
    QNetworkConfigurationManager m_mgr;
    QNetworkSession* m_session = nullptr;
    QList<NetworkInterface *> m_listInterface;
    QString m_pass = "1";
};

#endif // NETWORKMANAGER_H
