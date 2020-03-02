#include "networkparameterscontext.h"

NetworkInterfaceParameters::NetworkInterfaceParameters(QObject *parent):
   QObject (parent)
{
    m_ipAddress = QHostAddress::Any;
    m_gateWay = QHostAddress::Any;
    m_subnet = QHostAddress::Any;
}
NetworkInterfaceParameters::~NetworkInterfaceParameters()
{

}
NetworkParametersContext::NetworkParametersContext(QObject *parent):
    QObject (parent)
{

}
NetworkParametersContext::~NetworkParametersContext(){
    delete m_eth0;
    delete m_eth1;
}
