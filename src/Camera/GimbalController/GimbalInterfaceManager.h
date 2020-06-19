#ifndef GIMBALINTERFACEMANAGER_H
#define GIMBALINTERFACEMANAGER_H

#include <QObject>
#include "GimbalInterface.h"
#include "CM160/CM160Gimbal.h"
#include "Treron/TreronGimbal.h"
#include "Gremsey/GremseyGimbal.h"
#include "SBus/SBusGimbal.h"
class GimbalInterfaceManager : public QObject
{
    Q_OBJECT
public:
    enum class GIMBAL_TYPE{
        CM160=0,
        GREMSEY=1,
        TRERON=2,
        SBUS=3
    };
    explicit GimbalInterfaceManager(QObject *parent = nullptr);
    GimbalInterface* getGimbal(GIMBAL_TYPE type);
Q_SIGNALS:

public Q_SLOTS:
};

#endif // GIMBALINTERFACEMANAGER_H
