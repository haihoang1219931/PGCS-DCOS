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
#include <QVariantMap>
#include <QVariantList>
#include <stdio.h>
NetworkManager::NetworkManager(QObject *parent) : QObject(parent)
{
    qDBusRegisterMetaType<ConnectionSetting>();
    qDBusRegisterMetaType<QList<uint> >();
    qDBusRegisterMetaType<QList<QList<uint> > >();
    reloadListAccess();
    reloadListSetting();
    m_timerReloadAddress.setInterval(1000);
    m_timerReloadAddress.setSingleShot(true);
    connect(&m_timerReloadAddress,&QTimer::timeout,this,&NetworkManager::reloadAddresses);
}
void NetworkManager::reloadAddresses(){
    // get list info
    struct ifaddrs *ifaddr, *ifa;
    int s;
    char host[NI_MAXHOST];
    char subnet[NI_MAXHOST];
    char broadcast[NI_MAXHOST];
    QMap<QString,QMap<QString,QString>> addresses;
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
                QString hostString = QString::fromUtf8(host);
                QString hostInterface = QString::fromUtf8(ifa->ifa_name);
                QString hostGateway = QString::fromUtf8(broadcast).replace(".255",".1");
                addresses[hostInterface] ["Address"] = hostString;
                addresses[hostInterface] ["Subnet"] = QString::fromUtf8(subnet);
                addresses[hostInterface] ["Broadcast"] = QString::fromUtf8(broadcast);
                addresses[hostInterface] ["Gateway"] = hostGateway;
                if(hostString.length() >= 1 && hostString.at(0) == QChar('e')){
                    QString multicast = QString("echo ") + m_pass +
                            QString(" | sudo -S ip route add 224.0.0.0/4 dev ")+
                            hostInterface + " &";
                    printf("%s\r\n",multicast.toStdString().c_str());
                    system(multicast.toStdString().c_str());
                }else if(hostString.length() >= 1 && hostString.at(0) == QChar('w')){
                    QString internet =  QString("echo ") + m_pass +
                            QString(" | sudo -S ip route replace default via ")+
                            hostGateway + QString(" dev ") +
                            hostInterface + QString(" proto static &");
                    printf("%s\r\n",internet.toStdString().c_str());
                    system(internet.toStdString().c_str());
                }
            }
        }
    }

    freeifaddrs(ifaddr);
    for(int ifaceID = 0; ifaceID < m_listAccess.size(); ifaceID++){
        NetworkInterface *net = m_listAccess[ifaceID];
        if(addresses.keys().contains(net->name())){
            net->setActivated(true);
            net->setAddress(addresses[net->name()]["Address"]);
        }else{
            net->setActivated(false);
            net->setAddress("");
        }
    }
}
void NetworkManager::reloadListAccess(){
    QStringList listInterface;
    // get list interface ip
    m_listAccess.clear();
    QDBusInterface dbusInterface(
                "org.freedesktop.NetworkManager",
                "/org/freedesktop/NetworkManager",
                "org.freedesktop.NetworkManager",
                QDBusConnection::systemBus());
    QDBusReply<QList<QDBusObjectPath>> result = dbusInterface.call("GetDevices");
    Q_FOREACH (const QDBusObjectPath& connection, result.value()) {
        // get interface name
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    connection.path(),
                    "org.freedesktop.DBus.Properties",
                    QDBusConnection::systemBus());
        QDBusReply<QDBusVariant> deviceType = dbusInterface.call("Get",
                                                                 QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                                 QVariant::fromValue(QString("DeviceType")));
        QDBusReply<QDBusVariant> interfaceName = dbusInterface.call("Get",
                                                                    QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                                    QVariant::fromValue(QString("Interface")));
        QDBusReply<QDBusVariant> udiPath = dbusInterface.call("Get",
                                                                    QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                                    QVariant::fromValue(QString("Udi")));
        if (deviceType.isValid()) {
            QString name = interfaceName.value().variant().toString();
            int type =  deviceType.value().variant().toInt();
            QString udi =  udiPath.value().variant().toString();
//            printf("[%s][%d][%s][%s]\r\n",
//                   name.toStdString().c_str(),type,
//                   udi.toStdString().c_str(),
//                   connection.path().toStdString().c_str());
            if(type == 1){
                NetworkInterface *net = new NetworkInterface();
                net->setDevice(connection.path());
                net->setName(name);
                if(udi.contains("usb"))
                    net->setBearerTypeName("USB Ethernet");
                else if(udi.contains("pci") && udi.contains("pci"))
                    net->setBearerTypeName("PCI Ethernet");
                else
                    net->setBearerTypeName("Ethernet");
                getListSetting(net);
                m_listAccess.append(net);
                listInterface.append(net->bearerTypeName());
            }else if(type == 2){
                NetworkInterface *net = new NetworkInterface();
                net->setDevice(connection.path());
                net->setName(name);
                net->setBearerTypeName("Wi-Fi");
                getListAccess(net);
                m_listAccess.append(net);
                listInterface.append(net->bearerTypeName());
            }else if(type == 13){
                NetworkInterface *net = new NetworkInterface();
                net->setDevice(connection.path());
                net->setName(name);
                net->setBearerTypeName("Bridge");
                getListSetting(net);
                m_listAccess.append(net);
                listInterface.append(net->bearerTypeName());
            }
        }
    }
    reloadAddresses();
    // get list network
    Q_EMIT listAccessChanged();
}
void NetworkManager::reloadListSetting(){
    QStringList listNettypes;
    // get list interface ip
    m_listSetting.clear();
    QDBusInterface dbusInterface(
                "org.freedesktop.NetworkManager",
                "/org/freedesktop/NetworkManager",
                "org.freedesktop.NetworkManager",
                QDBusConnection::systemBus());
    QDBusReply<QList<QDBusObjectPath>> result = dbusInterface.call("GetDevices");
    Q_FOREACH (const QDBusObjectPath& connection, result.value()) {
        // get interface name
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    connection.path(),
                    "org.freedesktop.DBus.Properties",
                    QDBusConnection::systemBus());
        QDBusReply<QDBusVariant> deviceType = dbusInterface.call("Get",
                                                                 QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                                 QVariant::fromValue(QString("DeviceType")));
        QDBusReply<QDBusVariant> interfaceName = dbusInterface.call("Get",
                                                                    QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                                    QVariant::fromValue(QString("Interface")));
        QDBusReply<QDBusVariant> udiPath = dbusInterface.call("Get",
                                                                    QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                                    QVariant::fromValue(QString("Udi")));
        if (deviceType.isValid()) {
            QString name = interfaceName.value().variant().toString().replace("Disconnect: ","");
            int type =  deviceType.value().variant().toInt();
            QString udi =  udiPath.value().variant().toString();
//            printf("[%s][%d]\r\n",name.toStdString().c_str(),type);
            if(type == 1 && !listNettypes.contains("Ethernet")){
                NetworkInterface *net = new NetworkInterface();
                net->setDevice(connection.path());
                net->setName(name);
                net->setBearerTypeName("Ethernet");
                getListSetting(net);
                m_listSetting.append(net);
                listNettypes.append(net->bearerTypeName());
            }else if(type == 2 && !listNettypes.contains("Wi-Fi")){
                NetworkInterface *net = new NetworkInterface();
                net->setDevice(connection.path());
                net->setName(name);
                net->setBearerTypeName("Wi-Fi");
                getListSetting(net);
                m_listSetting.append(net);
                listNettypes.append(net->bearerTypeName());
            }else if(type == 13 && !listNettypes.contains("Bridge")){
                NetworkInterface *net = new NetworkInterface();
                net->setDevice(connection.path());
                net->setName(name);
                net->setBearerTypeName("Bridge");
                getListSetting(net);
                m_listSetting.append(net);
                listNettypes.append(net->bearerTypeName());
            }
        }
    }
    Q_EMIT listSettingChanged();
}
void NetworkManager::expose(){
    qmlRegisterType<NetworkInterface>("io.qdt.dev",1,0,"NetworkInterface");
    qmlRegisterType<NetworkInfo>("io.qdt.dev",1,0,"NetworkInfo");
    qmlRegisterType<NetworkManager>("io.qdt.dev",1,0,"NetworkManager");
    //    qmlRegisterType<QDBusArgument>("io.qdt.dev",1,0,"QDBusArgument");
}
void NetworkManager::connectNetwork(QString bearerTypeName, QString name,bool connect){
    printf("%s B[%s] N[%s] C[%s]\r\n",
           __func__,
           bearerTypeName.toStdString().c_str(),
           name.toStdString().c_str(),
           connect?"connect":"disconnect");
    bool connectResult = true;
    for(int ifaceID=0; ifaceID < m_listAccess.size(); ifaceID++){
        if(m_listAccess[ifaceID]->bearerTypeName() == bearerTypeName){
            for(int iSettingID = 0;
                iSettingID < m_listAccess[ifaceID]->getListNetwork().size();
                iSettingID ++){
                NetworkInfo* net = m_listAccess[ifaceID]->getListNetwork().at(iSettingID);
                if(net->name() == name){
                    if(bearerTypeName.contains("Ethernet")){
                        connectResult = connectSetting(net,connect);
                    }else if(bearerTypeName.contains("Wi-Fi")){
                        connectResult = connectAcessPoint(net,connect);
                    }
                    if(connectResult)
                        net->setActivated(connect);
                    break;
                }
            }
            if(connectResult){
                for(int iSettingID = 0;
                    iSettingID < m_listAccess[ifaceID]->getListNetwork().size();
                    iSettingID ++){
                    NetworkInfo* net = m_listAccess[ifaceID]->getListNetwork().at(iSettingID);
                    if(net->name() != name){
                        net->setActivated(false);
                    }
                }
            }
            break;
        }
    }
    m_timerReloadAddress.start();
}
bool NetworkManager::connectSetting(NetworkInfo* setting,bool connect){
    printf("%s setting device[%s] [%s]\r\n",
           connect?"Connect":"Disconnect",
           setting->device().toStdString().c_str(),
           setting->setting().toStdString().c_str());
    bool connectResult = true;
    if(!connect){
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    setting->device(),
                    "org.freedesktop.NetworkManager.Device",
                    QDBusConnection::systemBus());
        QDBusReply<void> result = dbusInterface.call("Disconnect");
        if (!result.isValid()) {
            qDebug() << QString("Error Disconnect connection: %1 %2").arg(result.error().name()).arg(result.error().message());
            connectResult = false;
        }else{
            qDebug() << QString("Disconnected");
        }

    }else{
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    "/org/freedesktop/NetworkManager",
                    "org.freedesktop.NetworkManager",
                    QDBusConnection::systemBus());
        QDBusReply<QDBusObjectPath> result = dbusInterface.call("ActivateConnection",
                                                                QVariant::fromValue(QDBusObjectPath(setting->setting())),
                                                                QVariant::fromValue(QDBusObjectPath(setting->device())),
                                                                QVariant::fromValue(QDBusObjectPath("/")));
        if (!result.isValid()) {
            qDebug() << QString("Error ActivateConnection connection: %1 %2").arg(result.error().name()).arg(result.error().message());
            connectResult = false;
        }else{
            qDebug() << QString("Connected: %1").arg(result.value().path());
        }
    }
    return connectResult;
}
bool NetworkManager::connectAcessPoint(NetworkInfo* accessPoint,bool connect){
    printf("%s accessPoint device[%s] [%s]\r\n",
           connect?"Connect":"Disconnect",
           accessPoint->device().toStdString().c_str(),
           accessPoint->accessPoint().toStdString().c_str());
    bool connectResult = true;
    if(!connect){
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    accessPoint->device(),
                    "org.freedesktop.NetworkManager.Device",
                    QDBusConnection::systemBus());
        QDBusReply<void> result = dbusInterface.call("Disconnect");
        if (!result.isValid()) {
            qDebug() << QString("Error Disconnected connection: %1 %2").arg(result.error().name()).arg(result.error().message());
            connectResult = false;
        }else{
            qDebug() << QString("Disconnected");
        }
    }else{
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    "/org/freedesktop/NetworkManager",
                    "org.freedesktop.NetworkManager",
                    QDBusConnection::systemBus());
        QDBusReply<QDBusObjectPath> result = dbusInterface.call("ActivateConnection",
                                                                QVariant::fromValue(QDBusObjectPath("/")),
                                                                QVariant::fromValue(QDBusObjectPath(accessPoint->device())),
                                                                QVariant::fromValue(QDBusObjectPath(accessPoint->accessPoint())));
        if (!result.isValid()) {
            qDebug() << QString("Error ActivateConnection connection: %1 %2").arg(result.error().name()).arg(result.error().message());
            connectResult = false;
            m_currentAccessPoint = accessPoint;
            Q_EMIT needWLANPass();

        }else{
            qDebug() << QString("Connected: %1").arg(result.value().path());
        }
    }
    return connectResult;
}
QString NetworkManager::convertIP(QString type,uint ipv4){
    QString result;
    if(type.contains("v4")){
        result = QHostAddress(qToBigEndian(ipv4)).toString();
    }
    return result;
}
void NetworkManager::insertWLANPass(QString pass){
    if(m_currentAccessPoint == nullptr) return;
    ConnectionSetting connection;
    // Build up the 'connection' Setting
    connection["802-11-wireless"]["security"] = "802-11-wireless-security";
    connection["802-11-wireless-security"]["key-mgmt"] = "wpa-psk";
    connection["802-11-wireless-security"]["psk"] = pass;
    QDBusInterface dbusInterface(
                "org.freedesktop.NetworkManager",
                "/org/freedesktop/NetworkManager",
                "org.freedesktop.NetworkManager",
                QDBusConnection::systemBus());
    QDBusReply<QDBusObjectPath> result = dbusInterface.call("AddAndActivateConnection",
                                                            QVariant::fromValue(connection),
                                                            QVariant::fromValue(QDBusObjectPath(m_currentAccessPoint->device())),
                                                            QVariant::fromValue(QDBusObjectPath(m_currentAccessPoint->accessPoint())));
    if (!result.isValid()) {
        qDebug() << QString("Error AddAndActivateConnection connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        Q_EMIT needWLANPass();
    }else{
        qDebug() << QString("Connected: %1").arg(result.value().path());
        m_currentAccessPoint->setActivated(true);
        m_currentAccessPoint = nullptr;
    }
}
void NetworkManager::deleteSetting(QString setting){
    QDBusInterface ifaceForSettings(
                "org.freedesktop.NetworkManager",
                setting,
                "org.freedesktop.NetworkManager.Settings.Connection",
                QDBusConnection::systemBus());
    QDBusReply<void> result = ifaceForSettings.call("Delete");
    if (!result.isValid()) {
        qDebug() << QString("Error Delete connection: %1 %2").arg(result.error().name()).arg(result.error().message());
        Q_EMIT needWLANPass();
    }else{
        qDebug() << QString("Deleted: %1").arg(setting);
    }
    reloadListSetting();
}
void NetworkManager::getConnectionSetting(NetworkInterface *interface,QString settingPath, NetworkInfo *setting){
    ConnectionSetting settings;
    QDBusInterface ifaceForSettings(
                "org.freedesktop.NetworkManager",
                settingPath,
                "org.freedesktop.NetworkManager.Settings.Connection",
                QDBusConnection::systemBus());
    QDBusReply<ConnectionSetting> result = ifaceForSettings.call("GetSettings");
    if (result.isValid()) {
        settings = result.value();
        QVariantMap connectionSettings = settings.value("connection");
        if(connectionSettings["type"].toString().contains("bridge")){
            setting->setBearerTypeName("Bridge");
        }else if(connectionSettings["type"].toString().contains("wireless")){
            setting->setBearerTypeName("Wi-Fi");
        }else if(connectionSettings["type"].toString().contains("ethernet")){
            setting->setBearerTypeName("Ethernet");
        }
        setting->setName(connectionSettings["id"].toString());
        setting->setSettingMap(settings);
    }else{

    }
}
QVariant NetworkManager::getConnectionSetting(QString settingPath){
    QVariant resultReturn;
    ConnectionSetting settings;
    QDBusInterface ifaceForSettings(
                "org.freedesktop.NetworkManager",
                settingPath,
                "org.freedesktop.NetworkManager.Settings.Connection",
                QDBusConnection::systemBus());
    QDBusMessage reply = ifaceForSettings.call("GetSettings");
    const QDBusArgument &dbusArg = reply.arguments().at( 0 ).value<QDBusArgument>();
    QMap<QString,QMap<QString,QVariant> > map;
    dbusArg >> map;
    // We're just printing out the data in a more user-friendly way here
    QVariantMap mapSetting;
    for( QString outer_key : map.keys() ){
        QMap<QString,QVariant> innerMap = map.value( outer_key );
//        qDebug() << "Key: " << outer_key;
        QVariantMap qmlInnerMap;
        for( QString inner_key : innerMap.keys() ){
//                        qDebug() << "    " << inner_key << ":" << innerMap.value( inner_key );
            QString dataTypeName(innerMap.value( inner_key ).typeName());
//                        qDebug() << "        dataTypeName "<< dataTypeName;
            if(dataTypeName == "QDBusArgument"){
                if(inner_key == "addresses"){
                    QList<QList<uint>> elems = qdbus_cast<QList<QList<uint>>>( innerMap.value( inner_key ) );
//                    qDebug() << "      " << elems;
                    QVariantList listAddress;
                    for(const QList<uint>& addresses: elems){
                        QVariantMap addr;
                        if(addresses.size()>=1){
                            addr["address"] = convertIP(outer_key,addresses[0]);
                        }
                        if(addresses.size()>=2){
                            if(addresses[1]>32){
                                addr["netmask"] = convertIP(outer_key,addresses[1]);
                            }else{
                                addr["netmask"] = addresses[1];
                            }
                        }else{
                            addr["netmask"] = 8;
                        }
                        if(addresses.size()>=3){
                            addr["gateway"] = convertIP(outer_key,addresses[2]);
                        }else{
                            addr["gateway"] = 0;
                        }
                        listAddress << addr;
                    }
                    qmlInnerMap["addresses"] = QVariant::fromValue(listAddress);
                }else if(inner_key == "dns"){
                    QList<uint> elems = qdbus_cast<QList<uint>>( innerMap.value( inner_key ) );
                    QVariantList listAddress;
                    for(const uint& address: elems){
                        listAddress << convertIP(outer_key,address);
                    }
                    qmlInnerMap[inner_key] = QVariant::fromValue(listAddress);
                }else if(inner_key == "dns-search"){
                    QList<QString> elems = qdbus_cast<QList<QString>>( innerMap.value( inner_key ) );
                    QVariantList listAddress;
                    for(const QString& address: elems){
                        listAddress << address;
                    }
                    qmlInnerMap[inner_key] = QVariant::fromValue(listAddress);
                }
            }else{
                if(inner_key == "mac-address"){
                    QByteArray mac = innerMap[inner_key].value<QByteArray>();
                    qDebug() << "mac-address: " << innerMap[inner_key];
                    QString macAddress;
                    for(int bID = 0; bID < mac.size(); bID++){
                        char macElement[10] ;
                        if(bID < mac.size()-1)
                            sprintf(macElement,"%02X:",static_cast<unsigned char>(mac.at(bID)));
                        else
                            sprintf(macElement,"%02X",static_cast<unsigned char>(mac.at(bID)));
                        macAddress+=QString::fromUtf8(macElement);
                    }
                    qmlInnerMap[inner_key] =  macAddress;
                }else{
                    qmlInnerMap[inner_key] = innerMap[inner_key];
                }


            }
        }
        mapSetting[outer_key] = QVariant(qmlInnerMap);
    }
    if(mapSetting["connection"].toMap()["type"].toString().contains("wireless")){
        // get secrets
        QDBusInterface ifaceForSettings(
                    "org.freedesktop.NetworkManager",
                    settingPath,
                    "org.freedesktop.NetworkManager.Settings.Connection",
                    QDBusConnection::systemBus());
        QDBusMessage reply = ifaceForSettings.call("GetSecrets",
                        QVariant::fromValue(QString("802-11-wireless-security")));
//        qDebug() << "Secrets: " << reply;
        const QDBusArgument &dbusArg = reply.arguments().at( 0 ).value<QDBusArgument>();
        QMap<QString,QMap<QString,QVariant> > map;
        dbusArg >> map;
        for( QString outer_key : map.keys() ){
            QMap<QString,QVariant> innerMap = map.value( outer_key );
            QVariantMap qmlInnerMap;
            if(mapSetting.contains(outer_key)){
                qmlInnerMap = mapSetting[outer_key].toMap();
            }
            for( QString inner_key : innerMap.keys() ){

                qmlInnerMap[inner_key] =  innerMap[inner_key];
            }
            mapSetting[outer_key] = QVariant(qmlInnerMap);
        }
    }
    resultReturn = QVariant::fromValue(mapSetting);
    return resultReturn;
}
void NetworkManager::getListAccess(NetworkInterface* interfaceName){
    printf("%s\r\n",__func__);
    // Get active connection
    QString activeConnectionID;
    {
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    interfaceName->device(),
                    "org.freedesktop.DBus.Properties",
                    QDBusConnection::systemBus());
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                             QVariant::fromValue(QString("ActiveConnection")));
        QString activeConnectionPath = result.value().variant().value<QDBusObjectPath>().path();
        if (!result.isValid()) {
            qDebug() << QString("Error get ActiveConnection: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            qDebug() << interfaceName->device() << ":" << activeConnectionPath;
        }
        if(activeConnectionPath != "/"){
            QDBusInterface dbusInterface(
                        "org.freedesktop.NetworkManager",
                        activeConnectionPath,
                        "org.freedesktop.DBus.Properties",
                        QDBusConnection::systemBus());
            QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                                 QVariant::fromValue(QString("org.freedesktop.NetworkManager.Connection.Active")),
                                                                 QVariant::fromValue(QString("Id")));
            if (!result.isValid()) {
                qDebug() << QString("Error Get Id: %1 %2").arg(result.error().name()).arg(result.error().message());
            }else{
                qDebug() << activeConnectionPath << ":" << result.value().variant().toString();
            }
            activeConnectionID = result.value().variant().toString();
        }
    }
    // Get list setting
    {
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    interfaceName->device(),
                    "org.freedesktop.NetworkManager.Device.Wireless",
                    QDBusConnection::systemBus());
        QDBusReply<QList<QDBusObjectPath>> result = dbusInterface.call("GetAllAccessPoints");
        if (!result.isValid()) {
            qDebug() << QString("Error GetAllAccessPoints: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            Q_FOREACH (const QDBusObjectPath& connection, result.value()) {
                //                    qDebug() << "AccessPoint: " << connection.path();
                NetworkInfo *accessPoint = new NetworkInfo();
                accessPoint->setBearerTypeName(interfaceName->bearerTypeName());
                accessPoint->setAccessPoint(connection.path());
                accessPoint->setDevice(interfaceName->device());
                getAccessPointInfo(accessPoint);
                if(activeConnectionID == accessPoint->name()){
                    accessPoint->setActivated(true);
                }
                if(accessPoint->name()!= ""){
                    interfaceName->append(accessPoint);
                }else{
                    delete accessPoint;
                }
            }
        }
    }
}
void NetworkManager::getListSetting(NetworkInterface *interface){
    printf("%s(%s)\r\n",__func__,
           interface->device().toStdString().c_str());
    if(interface == nullptr){
        return;
    }
    QDBusInterface dbusInterface(
                "org.freedesktop.NetworkManager",
                interface->device(),
                "org.freedesktop.DBus.Properties",
                QDBusConnection::systemBus());
    // Get active connection
    QString activeConnectionID;
    {
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.Device")),
                                                             QVariant::fromValue(QString("ActiveConnection")));
        QString activeConnectionPath = result.value().variant().value<QDBusObjectPath>().path();
        if(activeConnectionPath!="/")
        {
            QDBusInterface dbusInterface(
                        "org.freedesktop.NetworkManager",
                        activeConnectionPath,
                        "org.freedesktop.DBus.Properties",
                        QDBusConnection::systemBus());
            QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                                 QVariant::fromValue(QString("org.freedesktop.NetworkManager.Connection.Active")),
                                                                 QVariant::fromValue(QString("Id")));
            if (!result.isValid()) {
                qDebug() << QString("Error Get Id: %1 %2").arg(result.error().name()).arg(result.error().message());
            }else{
                qDebug() << activeConnectionPath << ":" << result.value().variant().toString();
            }
            activeConnectionID = result.value().variant().toString();
        }
    }
    // Get list setting
    {
        QDBusInterface dbusInterface(
                    "org.freedesktop.NetworkManager",
                    "/org/freedesktop/NetworkManager/Settings",
                    "org.freedesktop.DBus.Properties",
                    QDBusConnection::systemBus());
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.Settings")),
                                                             QVariant::fromValue(QString("Connections")));
        if (!result.isValid()) {
            qDebug() << QString("Error get connections: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{

        }
        const QDBusArgument & dbusArgs = result.value().variant().value<QDBusArgument>();
        QDBusObjectPath path;
        dbusArgs.beginArray();
        while (!dbusArgs.atEnd())
        {

            dbusArgs >> path;
            NetworkInfo *setting = new NetworkInfo();
            setting->setDevice(interface->device());
            setting->setSetting(path.path());
            getConnectionSetting(interface,path.path(),setting);
            if(activeConnectionID == setting->name()){
                setting->setActivated(true);
            }
            if(interface->bearerTypeName().contains(setting->bearerTypeName())){
                setting->setBearerTypeName(interface->bearerTypeName());
                interface->append(setting);
            }
        }
        dbusArgs.endArray();
    }
}
void NetworkManager::getAccessPointInfo(NetworkInfo* accessPoint){
    QDBusInterface dbusInterface(
                "org.freedesktop.NetworkManager",
                accessPoint->accessPoint(),
                "org.freedesktop.DBus.Properties",
                QDBusConnection::systemBus());
    // Get Ssid
    {
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
                                                             QVariant::fromValue(QString("Ssid")));
        if (!result.isValid()) {
            qDebug() << QString("Error Get Ssid: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
//            qDebug() << "Ssid: " << result.value().variant().toString();
            accessPoint->setName(result.value().variant().toString());
        }
    }
    // Get frequency
    {
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
                                                             QVariant::fromValue(QString("Frequency")));
        if (!result.isValid()) {
            qDebug() << QString("Error Get Frequency: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            //            qDebug() << "Frequency: " << result.value().variant().toInt();
            accessPoint->setFrequency(result.value().variant().toInt());
        }
    }
    // Get password access
    {
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
                                                             QVariant::fromValue(QString("Strength")));
        if (!result.isValid()) {
            qDebug() << QString("Error Get Strength: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            //            qDebug() << "Strength: " << result.value().variant().toInt();
            accessPoint->setStrength(result.value().variant().toInt());
        }
    }
    // Has pass
    {
        QDBusReply<QDBusVariant> result = dbusInterface.call("Get",
                                                             QVariant::fromValue(QString("org.freedesktop.NetworkManager.AccessPoint")),
                                                             QVariant::fromValue(QString("Flags")));
        if (!result.isValid()) {
            qDebug() << QString("Error Get Flags: %1 %2").arg(result.error().name()).arg(result.error().message());
        }else{
            //            qDebug() << "Flags: " << result.value().variant().toInt();
            accessPoint->setHasPass(result.value().variant().toInt() != 0);
        }
    }
}
/*
 * dbus-send --system --print-reply \
            --dest=org.freedesktop.NetworkManager \
            /org/freedesktop/NetworkManager/Settings/18 \
            org.freedesktop.NetworkManager.Settings.Connection.GetSettings
method return time=1597035632.794043 sender=:1.6 -> destination=:1.765 serial=408774 reply_serial=2
   array [
      dict entry(
         string "802-3-ethernet"
         array [
            dict entry(
               string "duplex"
               variant                   string "full"
            )
            dict entry(
               string "mac-address-blacklist"
               variant                   array [
                  ]
            )
            dict entry(
               string "s390-options"
               variant                   array [
                  ]
            )
         ]
      )
      dict entry(
         string "connection"
         array [
            dict entry(
               string "id"
               variant                   string "192.168.0.220"
            )
            dict entry(
               string "uuid"
               variant                   string "d57b5279-ee2c-4a33-9aec-f8e10bae3992"
            )
            dict entry(
               string "type"
               variant                   string "802-3-ethernet"
            )
            dict entry(
               string "permissions"
               variant                   array [
                  ]
            )
            dict entry(
               string "timestamp"
               variant                   uint64 1592792919
            )
            dict entry(
               string "secondaries"
               variant                   array [
                  ]
            )
         ]
      )
      dict entry(
         string "ipv6"
         array [
            dict entry(
               string "method"
               variant                   string "auto"
            )
            dict entry(
               string "dns"
               variant                   array [
                  ]
            )
            dict entry(
               string "dns-search"
               variant                   array [
                  ]
            )
            dict entry(
               string "addresses"
               variant                   array [
                  ]
            )
            dict entry(
               string "routes"
               variant                   array [
                  ]
            )
            dict entry(
               string "ip6-privacy"
               variant                   int32 0
            )
            dict entry(
               string "address-data"
               variant                   array [
                  ]
            )
            dict entry(
               string "route-data"
               variant                   array [
                  ]
            )
         ]
      )
      dict entry(
         string "ipv4"
         array [
            dict entry(
               string "method"
               variant                   string "manual"
            )
            dict entry(
               string "dns"
               variant                   array [
                  ]
            )
            dict entry(
               string "dns-search"
               variant                   array [
                  ]
            )
            dict entry(
               string "addresses"
               variant                   array [
                     array [
                        uint32 73705664
                        uint32 24
                        uint32 23374016
                     ]
                     array [
                        uint32 3724607498
                        uint32 8
                        uint32 0
                     ]
                  ]
            )
            dict entry(
               string "gateway"
               variant                   string "192.168.100.1"
            )
            dict entry(
               string "routes"
               variant                   array [
                  ]
            )
            dict entry(
               string "address-data"
               variant                   array [
                     array [
                        dict entry(
                           string "address"
                           variant                               string "192.168.100.4"
                        )
                        dict entry(
                           string "prefix"
                           variant                               uint32 24
                        )
                     ]
                     array [
                        dict entry(
                           string "address"
                           variant                               string "10.0.1.222"
                        )
                        dict entry(
                           string "prefix"
                           variant                               uint32 8
                        )
                     ]
                  ]
            )
            dict entry(
               string "route-data"
               variant                   array [
                  ]
            )
         ]
      )
   ]


dbus-send --system --print-reply             --dest=org.freedesktop.NetworkManager             /org/freedesktop/NetworkManager/Settings/3             org.freedesktop.NetworkManager.Settings.Connection.GetSettings
method return time=1597070278.993978 sender=:1.10 -> destination=:1.670 serial=339655 reply_serial=2
   array [
      dict entry(
         string "connection"
         array [
            dict entry(
               string "id"
               variant                   string "Hoanghait3"
            )
            dict entry(
               string "uuid"
               variant                   string "f8e98ee9-54e6-4c83-bbdd-70754cb8584a"
            )
            dict entry(
               string "type"
               variant                   string "802-11-wireless"
            )
            dict entry(
               string "permissions"
               variant                   array [
                  ]
            )
            dict entry(
               string "timestamp"
               variant                   uint64 1597070201
            )
            dict entry(
               string "secondaries"
               variant                   array [
                  ]
            )
         ]
      )
      dict entry(
         string "802-11-wireless"
         array [
            dict entry(
               string "ssid"
               variant                   array of bytes "Hoanghait3"
            )
            dict entry(
               string "mode"
               variant                   string "infrastructure"
            )
            dict entry(
               string "mac-address"
               variant                   array of bytes [
                     a8 6b ad 53 56 3f
                  ]
            )
            dict entry(
               string "mac-address-blacklist"
               variant                   array [
                  ]
            )
            dict entry(
               string "seen-bssids"
               variant                   array [
                     string "C4:71:54:0B:F7:42"
                  ]
            )
            dict entry(
               string "security"
               variant                   string "802-11-wireless-security"
            )
         ]
      )
      dict entry(
         string "802-11-wireless-security"
         array [
            dict entry(
               string "key-mgmt"
               variant                   string "wpa-psk"
            )
            dict entry(
               string "auth-alg"
               variant                   string "open"
            )
            dict entry(
               string "proto"
               variant                   array [
                  ]
            )
            dict entry(
               string "pairwise"
               variant                   array [
                  ]
            )
            dict entry(
               string "group"
               variant                   array [
                  ]
            )
         ]
      )
      dict entry(
         string "ipv4"
         array [
            dict entry(
               string "method"
               variant                   string "auto"
            )
            dict entry(
               string "dns"
               variant                   array [
                  ]
            )
            dict entry(
               string "dns-search"
               variant                   array [
                  ]
            )
            dict entry(
               string "addresses"
               variant                   array [
                  ]
            )
            dict entry(
               string "routes"
               variant                   array [
                  ]
            )
            dict entry(
               string "address-data"
               variant                   array [
                  ]
            )
            dict entry(
               string "route-data"
               variant                   array [
                  ]
            )
         ]
      )
      dict entry(
         string "ipv6"
         array [
            dict entry(
               string "method"
               variant                   string "auto"
            )
            dict entry(
               string "dns"
               variant                   array [
                  ]
            )
            dict entry(
               string "dns-search"
               variant                   array [
                  ]
            )
            dict entry(
               string "addresses"
               variant                   array [
                  ]
            )
            dict entry(
               string "routes"
               variant                   array [
                  ]
            )
            dict entry(
               string "address-data"
               variant                   array [
                  ]
            )
            dict entry(
               string "route-data"
               variant                   array [
                  ]
            )
         ]
      )
   ]


*/
