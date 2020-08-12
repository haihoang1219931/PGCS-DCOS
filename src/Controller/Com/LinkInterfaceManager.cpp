#include "LinkInterfaceManager.h"
#include "TCPLink.h"
#include "UDPLink.h"
#include "UDPMulticastLink.h"
#include "SerialLink.h"
LinkInterfaceManager::LinkInterfaceManager(QObject *parent) : QObject(parent)
{

}
LinkInterface* LinkInterfaceManager::linkForAPConnection(CONNECTION_TYPE type){
    switch (type) {
        case CONNECTION_TYPE::MAV_TCP:
            printf("Create TCP Link\r\n");
            return new TCPLink();
        case CONNECTION_TYPE::MAV_RAGAS:
            printf("Create UDPMulticast Link\r\n");
            return new UDPMulticastLink();
        case CONNECTION_TYPE::MAV_SERIAL:
            printf("Create Serial Link\r\n");
            return new SerialLink();
        case CONNECTION_TYPE::MAV_UDP:
            printf("Create UDP Link\r\n");
            return new UDPLink();
        default:
            return NULL;
    }
}
