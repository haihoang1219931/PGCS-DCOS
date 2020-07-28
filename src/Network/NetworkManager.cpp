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
NetworkManager::NetworkManager(QObject *parent) : QObject(parent)
{
    reload();
}
void NetworkManager::reload(){
    // get list network
    m_listNetwork.clear();
    QList<QNetworkConfiguration> configs = m_mgr.allConfigurations();
    for(int i=0; i< configs.size(); i++){
        QNetworkConfiguration tmp = configs[i];
        if(tmp.identifier().contains("NetworkManager")){
            NetworkInfo *net = new NetworkInfo();
            net->setConfig(tmp);
            m_listNetwork.append(net);
        }
    }
    Q_EMIT listNetworkChanged();
    // get list interface ip
    m_listInterface.clear();
    struct ifaddrs *ifaddr, *ifa;
    int s;
    char host[NI_MAXHOST];
    if (getifaddrs(&ifaddr) == -1)
    {
        perror("getifaddrs");
    }else{
        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
        {
            if (ifa->ifa_addr == NULL)
                continue;
            s = getnameinfo(ifa->ifa_addr,sizeof(struct sockaddr_in),host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
            if(ifa->ifa_addr->sa_family==AF_INET)
            {
                if (s != 0)
                {
//                    printf("getnameinfo() failed: %s\n", gai_strerror(s));
                }else{
//                    printf("\tInterface : <%s>\n",ifa->ifa_name );
//                    printf("\t  Address : <%s>\n", host);
                    if(QString::fromUtf8(ifa->ifa_name) != "lo"){
                        NetworkInfo *net = new NetworkInfo();
                        net->setName(QString::fromUtf8(ifa->ifa_name));
                        net->setAddress(QString::fromUtf8(host));
                        m_listInterface.append(net);
                    }
                }
            }
        }
    }
    freeifaddrs(ifaddr);
    Q_EMIT listInterfaceChanged();
}
void NetworkManager::expose(){
    qmlRegisterType<NetworkInfo>("io.qdt.dev",1,0,"NetworkInfo");
    qmlRegisterType<NetworkManager>("io.qdt.dev",1,0,"NetworkManager");
}
void NetworkManager::connectNetwork(QString name,bool connect){
    QDBusInterface interface(
        "org.freedesktop.NetworkManager",
        "/org/freedesktop/NetworkManager/Devices/0",
        "org.freedesktop.NetworkManager.Device",
        QDBusConnection::systemBus());
    QNetworkAccessManager agr;
    for(int i=0; i< m_listNetwork.size(); i++){
        NetworkInfo* tmp = m_listNetwork[i];
        if(tmp->config().name() == name){
            QNetworkConfiguration config = tmp->config();
            printf("%s [%s] [%s] [%s] [%d]\r\n",connect?
                       "Connect":"Disconnect",
                   config.identifier().toStdString().c_str(),
                   config.bearerTypeName().toStdString().c_str(),
                   config.name().toStdString().c_str(),
                   config.type());
            const bool canStartIAP = (m_mgr.capabilities()
                                      & QNetworkConfigurationManager::CanStartAndStopInterfaces);
            if (!config.isValid() || (!canStartIAP && config.state() != QNetworkConfiguration::Active)) {
                printf("No Access Point found\r\n");
            }else{
                printf("Access Point found\r\n");
            }
            if(connect){
                QDBusReply<QDBusObjectPath> result = interface.call("ActivateConnection",QVariant::fromValue(config.identifier()));
                if (!result.isValid()) {
                    qDebug() << QString("Error Connect: %1 %2").arg(result.error().name()).arg(result.error().message());
                } else {
                    qDebug() << QString("Connect: %1").arg(result.value().path());
                }
            }else{
                QDBusReply<QDBusObjectPath> result = interface.call("Disconnect");
                if (!result.isValid()) {
                    qDebug() << QString("Error Disconnect: %1 %2").arg(result.error().name()).arg(result.error().message());
                } else {
                    qDebug() << QString("Disconnect: %1").arg(result.value().path());
                }
            }

            break;
        }
    }
}
