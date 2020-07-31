#include "NetworkManager.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <QQmlApplicationEngine>
#include <QVariant>
#include <stdio.h>
NetworkManager::NetworkManager(QObject *parent) : QObject(parent)
{
    qDBusRegisterMetaType<Connection>();
    qDBusRegisterMetaType<QList<uint> >();
    qDBusRegisterMetaType<QList<QList<uint> > >();
    reload();
}
void NetworkManager::reload(){
    // get list interface ip
    m_listInterface.clear();
    QDBusInterface interface(
        "org.freedesktop.NetworkManager",
        "/org/freedesktop/NetworkManager",
        "org.freedesktop.NetworkManager",
        QDBusConnection::systemBus());
    QDBusReply<QList<QDBusObjectPath>> result = interface.call("GetDevices");
    Q_FOREACH (const QDBusObjectPath& connection, result.value()) {
            qDebug() << connection.path();
            // get interface name
            QDBusInterface interface(
                "org.freedesktop.NetworkManager",
                connection.path(),
                "org.freedesktop.DBus.Properties",
                QDBusConnection::systemBus());
            QDBusReply<QDBusVariant> interfaceName = interface.call("Get",
                QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                QVariant::fromValue(QString("Interface")));
            if (interfaceName.isValid()) {
                QString name = interfaceName.value().variant().toString().replace("Disconnect: ","");
                if(name != "lo"){
                    NetworkInterface *net = new NetworkInterface();
                    net->setDevice(connection.path());
                    net->setName(name);
                    if(net->name().length() >= 1 && net->name().at(0) == QChar('e')){
                        net->setBearerTypeName("Ethernet");
                        getListSettings(net);
                        m_listInterface.append(net);
                    }else if(net->name().length() >= 1 && net->name().at(0) == QChar('w')){
                        net->setBearerTypeName("WLAN");
                        getListSettings(net);
                        m_listInterface.append(net);
                    }else{
                        delete net;
                    }

                }
            }

    }
    // get list info
    struct ifaddrs *ifaddr, *ifa;
    int s;
    char host[NI_MAXHOST];
    char subnet[NI_MAXHOST];
    char broadcast[NI_MAXHOST];
    if (getifaddrs(&ifaddr) == -1)
    {
        perror("getifaddrs");
    }else{
        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
        {
            if (ifa->ifa_addr == NULL)
                continue;
            s = getnameinfo(ifa->ifa_addr,sizeof(struct sockaddr_in),host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
            if(ifa->ifa_addr->sa_family==AF_INET && s == 0)
            {
                getnameinfo(ifa->ifa_ifu.ifu_broadaddr,sizeof(struct sockaddr_in),broadcast, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
                getnameinfo(ifa->ifa_netmask,sizeof(struct sockaddr_in),subnet, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
                for(int ifaceID = 0; ifaceID < m_listInterface.size(); ifaceID++){
                    NetworkInterface *net = m_listInterface[ifaceID];
                    if(QString::fromUtf8(ifa->ifa_name) == net->name()){
                        net->setAddress(QString::fromUtf8(host));
                        net->setSubnet(QString::fromUtf8(subnet));
                        net->setBroadcast(QString::fromUtf8(broadcast));
                        net->setGateway(net->broadcast().replace(".255",".1"));
                        if(net->name().length() >= 1 && net->name().at(0) == QChar('e')){
                            QString multicast = QString("echo ") + m_pass +
                                    QString(" | sudo -S ip route add 224.0.0.0/4 dev ")+
                                    net->name() + " &";
                            system(multicast.toStdString().c_str());
                        }else if(net->name().length() >= 1 && net->name().at(0) == QChar('w')){
                            QString internet =  QString("echo ") + m_pass +
                                    QString(" | sudo -S ip route replace default via ")+
                                    net->gateway() + QString(" dev ") +
                                    net->name() + QString(" proto static &");
                            system(internet.toStdString().c_str());
                        }
                        break;
                    }
                }
            }
        }
    }
    freeifaddrs(ifaddr);

    // get list network
    Q_EMIT listInterfaceChanged();
}
void NetworkManager::expose(){
    qmlRegisterType<NetworkInterface>("io.qdt.dev",1,0,"NetworkInterface");
    qmlRegisterType<NetworkInfo>("io.qdt.dev",1,0,"NetworkInfo");
    qmlRegisterType<NetworkManager>("io.qdt.dev",1,0,"NetworkManager");
}
void NetworkManager::connectNetwork(QString name,bool connect){
    for(int ifaceID=0; ifaceID < m_listInterface.size(); ifaceID++){
        for(int iSettingID = 0;
            iSettingID < m_listInterface[ifaceID]->getListNetwork().size();
            iSettingID ++){
            NetworkInfo* net = m_listInterface[ifaceID]->getListNetwork().at(iSettingID);
            if(net->name() == name){
                if(!connect){
                    QDBusInterface interface(
                        "org.freedesktop.NetworkManager",
                        net->device(),
                        "org.freedesktop.NetworkManager.Device",
                        QDBusConnection::systemBus());
                    QDBusReply<void> result = interface.call("Disconnect");
                    if (!result.isValid()) {
                        qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
                    }else{
                        net->setActivated(connect);
                        qDebug() << QString("Disconnected");
                    }

                }else{
                    QDBusInterface interface(
                        "org.freedesktop.NetworkManager",
                        "/org/freedesktop/NetworkManager",
                        "org.freedesktop.NetworkManager",
                        QDBusConnection::systemBus());
                    QDBusReply<QDBusObjectPath> result = interface.call("ActivateConnection",
                        QVariant::fromValue(QDBusObjectPath(net->setting())),
                        QVariant::fromValue(QDBusObjectPath(net->device())),
                        QVariant::fromValue(QDBusObjectPath("/")));
                    if (!result.isValid()) {
                        qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
                    }else{
                        net->setActivated(connect);
                        qDebug() << QString("Connected: %1").arg(result.value().path());
                    }

                }
            }
        }
    }
}
void NetworkManager::getConnectionSetting(QString settingPath, NetworkInfo *setting){
    Connection settings;
    QDBusInterface ifaceForSettings(
        "org.freedesktop.NetworkManager",
        settingPath,
        "org.freedesktop.NetworkManager.Settings.Connection",
        QDBusConnection::systemBus());
    QDBusReply<Connection> result = ifaceForSettings.call("GetSettings");
    if (result.isValid()) {
        settings = result.value();
        QVariantMap connectionSettings = settings.value("connection");
        setting->setName(settings.value("connection").value("id").toString());
        setting->setMode(settings.value("ipv4").value("method").toString());
    }
}
void NetworkManager::getListSettings(NetworkInterface *interfaceName){
    if(interfaceName == nullptr){
        return;
    }
    if(interfaceName->bearerTypeName() == "Ethernet"){
        QDBusInterface interface(
            "org.freedesktop.NetworkManager",
            interfaceName->device(),
            "org.freedesktop.DBus.Properties",
            QDBusConnection::systemBus());
        QDBusReply<QDBusVariant> result = interface.call("Get",
            QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
            QVariant::fromValue(QString("AvailableConnections")));
        const QDBusArgument & dbusArgs = result.value().variant().value<QDBusArgument>();
        QDBusObjectPath path;
        dbusArgs.beginArray();
        while (!dbusArgs.atEnd())
        {
            dbusArgs >> path;
            NetworkInfo *setting = new NetworkInfo();
            setting->setBearerTypeName(interfaceName->bearerTypeName());
            setting->setDevice(interfaceName->device());
            setting->setSetting(path.path());
            getConnectionSetting(path.path(),setting);
            interfaceName->append(setting);
        }
        dbusArgs.endArray();
    }else if(interfaceName->bearerTypeName() == "WLAN"){
        QDBusInterface interface(
            "org.freedesktop.NetworkManager",
            interfaceName->device(),
            "org.freedesktop.NetworkManager.Device.Wireless",
            QDBusConnection::systemBus());
        QDBusReply<QList<QDBusObjectPath>> result = interface.call("GetAllAccessPoints");
        if (!result.isValid()) {
            qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            Q_FOREACH (const QDBusObjectPath& connection, result.value()) {
                    qDebug() << "AccessPoint: " << connection.path();
                    NetworkInfo *accessPoint = new NetworkInfo();
                    accessPoint->setBearerTypeName(interfaceName->bearerTypeName());
                    accessPoint->setAccessPoint(connection.path());
                    getAccessPointInfo(accessPoint);
                    interfaceName->append(accessPoint);
            }
        }
    }
}
void NetworkManager::getAccessPointInfo(NetworkInfo* accessPoint){
    QDBusInterface interface(
        "org.freedesktop.NetworkManager",
        accessPoint->accessPoint(),
        "org.freedesktop.DBus.Properties",
        QDBusConnection::systemBus());
    // Get Ssid
    {
        QDBusReply<QDBusVariant> result = interface.call("Get",
           QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
           QVariant::fromValue(QString("Ssid")));
        if (!result.isValid()) {
            qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            qDebug() << "Ssid: " << result.value().variant().toString();
            accessPoint->setName(result.value().variant().toString());
        }
    }
    // Get frequency
    {
        QDBusReply<QDBusVariant> result = interface.call("Get",
           QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
           QVariant::fromValue(QString("Frequency")));
        if (!result.isValid()) {
            qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            qDebug() << "Frequency: " << result.value().variant().toInt();
            accessPoint->setFrequency(result.value().variant().toInt());
        }
    }
    // Get password access
    {
        QDBusReply<QDBusVariant> result = interface.call("Get",
           QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
           QVariant::fromValue(QString("Strength")));
        if (!result.isValid()) {
            qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            qDebug() << "Strength: " << result.value().variant().toInt();
            accessPoint->setStrength(result.value().variant().toInt());
        }
    }
    // Has pass
    {
        QDBusReply<QDBusVariant> result = interface.call("Get",
           QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
           QVariant::fromValue(QString("Flags")));
        if (!result.isValid()) {
            qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            qDebug() << "Flags: " << result.value().variant().toInt();
            accessPoint->setHasPass(result.value().variant().toInt() != 0);
        }
    }
}
