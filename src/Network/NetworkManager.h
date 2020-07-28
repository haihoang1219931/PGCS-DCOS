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
class NetworkManager : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<NetworkInfo> listNetwork READ listNetwork NOTIFY listNetworkChanged)
    Q_PROPERTY(QQmlListProperty<NetworkInfo> listInterface READ listInterface NOTIFY listInterfaceChanged)
public:
    explicit NetworkManager(QObject *parent = nullptr);
    QQmlListProperty<NetworkInfo> listNetwork()
    {
        return QQmlListProperty<NetworkInfo>(this, m_listNetwork);
    }
    QQmlListProperty<NetworkInfo> listInterface()
    {
        return QQmlListProperty<NetworkInfo>(this, m_listInterface);
    }
    Q_INVOKABLE void reload();
    static void expose();
public:
    Q_INVOKABLE void connectNetwork(QString name,bool connect);
Q_SIGNALS:
    void listNetworkChanged();
    void listInterfaceChanged();
public Q_SLOTS:
private:
    QNetworkConfigurationManager m_mgr;
    QNetworkSession* m_session = nullptr;
    QList<NetworkInfo *> m_listNetwork;
    QList<NetworkInfo *> m_listInterface;
};

#endif // NETWORKMANAGER_H
