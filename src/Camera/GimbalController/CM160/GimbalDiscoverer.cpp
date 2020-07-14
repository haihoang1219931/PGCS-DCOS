#include "GimbalDiscoverer.h"

GimbalDiscoverer::GimbalDiscoverer(QObject *parent) :
    QObject(parent)
{
    qDebug () << "Gimbal discover created";
    socket = new UdpSenderListener("255.255.255.255", UAVV_DISCOVER_PORT);
    socket->Open();
    connect(socket,SIGNAL(onNewMessageBroadcast()),this,SLOT(handleNewMessageRecieved()));
    //RequestDiscover();
}
GimbalDiscoverer::~GimbalDiscoverer(){

}
void GimbalDiscoverer::requestDiscover()
{
//    lstGimbal.clear();
    QByteArray msg;
    msg.append('S');
    msg.append('L');
    msg.append('D');
    msg.append('I');
    msg.append('S');
    msg.append('C');
    msg.append('O');
    msg.append('V');
    msg.append('E');
    msg.append('R');
    socket->Send(msg);
    printf("Send discoverMessage: %s\r\n",msg.data());
}
void GimbalDiscoverer::Dispose()
{
    socket->Dispose();
}
QVariantMap GimbalDiscoverer::GetListGimbal(){
    QVariantMap result;
    return result;
}

void GimbalDiscoverer::handleNewMessageRecieved()
{
    while (socket->msgRecBuff.size() > 0)
    {
        UdpPayload message = socket->GetMessage();
        DiscoverPacket discoverPacket;
        if (DiscoverPacket::TryParse(message.Data.data(),message.Data.size(),&discoverPacket))
        {
            Q_EMIT newDiscoverPacketReceived(
                        QString::fromStdString(discoverPacket.Name)
                        ,QString::fromStdString(discoverPacket.IPAddress)
                        ,QString::fromStdString(discoverPacket.VideoAddress),
                        discoverPacket.VideoPort,
                        discoverPacket.ComsPort,
                        discoverPacket.ComsPort+1);
        }else{

        }
    }
}


