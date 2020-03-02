#ifndef GIMBALINTERFACECONTEXT_H
#define GIMBALINTERFACECONTEXT_H

#include <QQuickItem>
#include "gimbalcontext.h"
#include "networkparameterscontext.h"
class GimbalInterfaceContext : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString name READ name)
    Q_PROPERTY(GimbalContext* gimbal READ gimbal)
    Q_PROPERTY(NetworkParametersContext* network READ network)
public:
    QString m_name;
    GimbalContext* m_gimbal;
    NetworkParametersContext* m_network;
    GimbalInterfaceContext(QObject*parent = 0);
    virtual ~GimbalInterfaceContext();
    QString name(){
      return m_name;
    }
    GimbalContext* gimbal(){
        return m_gimbal;
    }

    NetworkParametersContext* network(){
        return m_network;
    }
};

#endif // GIMBALINTERFACECONTEXT_H
