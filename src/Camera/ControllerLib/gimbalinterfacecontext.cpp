#include "gimbalinterfacecontext.h"

GimbalInterfaceContext::GimbalInterfaceContext(QObject* parent):
    QObject(parent)
{
    m_gimbal = new GimbalContext();
    m_network = new NetworkParametersContext();
}
GimbalInterfaceContext::~GimbalInterfaceContext(){
    delete m_gimbal;
    delete m_network;
}
