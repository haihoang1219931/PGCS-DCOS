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
                    }else if(net->name().length() >= 1 && net->name().at(0) == QChar('w')){
                        net->setBearerTypeName("WLAN");
                    }else{
                        net->setBearerTypeName("Unknown");
                    }
                    m_listInterface.append(net);
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
//                        printf("\tInterface : <%s>\n",net->name().toStdString().c_str() );
//                        printf("\tBearTypeN : <%s>\n",net->bearerTypeName().toStdString().c_str() );
//                        printf("\t  Address : <%s>\n", net->address().toStdString().c_str());
//                        printf("\tBroadcast : <%s>\n", net->broadcast().toStdString().c_str());
//                        printf("\t   Subnet : <%s>\n", net->subnet().toStdString().c_str());
//                        printf("\t  Gateway : <%s>\n", net->gateway().toStdString().c_str());
                        break;
                    }
                }
            }
        }
    }
    freeifaddrs(ifaddr);

    // get list network
    QList<QNetworkConfiguration> configs = m_mgr.allConfigurations();
    for(int i=0; i< configs.size(); i++){
        QNetworkConfiguration tmp = configs[i];
        if(tmp.identifier().contains("NetworkManager")){
            NetworkInfo *net = new NetworkInfo();
            net->setConfig(tmp);
            for(int ifaceID = 0; ifaceID < m_listInterface.size(); ifaceID ++){
//                printf("[%s] vs [%s]\r\n",
//                       m_listInterface[ifaceID]->bearerTypeName().toStdString().c_str(),
//                       net->bearerTypeName().toStdString().c_str());
                if(m_listInterface[ifaceID]->bearerTypeName() ==
                        net->bearerTypeName()){                    
                    net->setDevice(m_listInterface[ifaceID]->device());
                    m_listInterface[ifaceID]->append(net);
                }
            }
        }
    }

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
                        qDebug() << QString("Disconnected");
                    }
                    net->setActivated(connect);
                }else{
                    QDBusInterface interface(
                        "org.freedesktop.NetworkManager",
                        "/org/freedesktop/NetworkManager",
                        "org.freedesktop.NetworkManager",
                        QDBusConnection::systemBus());
                    printf("Setting:%s\r\n",net->setting().toStdString().c_str());
                    printf("Device :%s\r\n",net->device().toStdString().c_str());
                    QDBusReply<QDBusObjectPath> result = interface.call("ActivateConnection",
                        QVariant::fromValue(QDBusObjectPath(net->setting())),
                        QVariant::fromValue(QDBusObjectPath(net->device())),
                        QVariant::fromValue(QDBusObjectPath("/")));
                    if (!result.isValid()) {
                        qDebug() << QString("Error adding connection: %1 %2").arg(result.error().name()).arg(result.error().message());
                    }else{
                        qDebug() << QString("Connected: %1").arg(result.value().path());
                    }
                    net->setActivated(connect);
                }
            }
        }
    }
}
