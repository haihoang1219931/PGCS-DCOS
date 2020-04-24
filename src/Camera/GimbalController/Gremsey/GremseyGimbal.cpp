#include "GremseyGimbal.h"
#include "GimbalControl.h"
#include "TCPClient.h"
GremseyGimbal::GremseyGimbal(GimbalInterface *parent) : GimbalInterface(parent)
{

}
void GremseyGimbal::connectToGimbal(Config* config){
    if(config == nullptr)
        return;
    m_config = config;
    m_gimbal = new GRGimbalController();
    QVariantMap dataMap = m_config->getData().toMap();
    m_gimbal->setupTCP(dataMap["CAM_CONTROL_IP"].toString(),
                        dataMap["CAM_CONTROL_IN"].toInt());
}
void GremseyGimbal::disconnectGimbal(){
    if(m_gimbal==nullptr)
        return;
}
