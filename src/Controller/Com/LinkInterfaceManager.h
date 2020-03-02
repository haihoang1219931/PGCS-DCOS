#ifndef LINKINTERFACEMANAGER_H
#define LINKINTERFACEMANAGER_H

#include <QObject>
#include "LinkInterface.h"
class LinkInterfaceManager : public QObject
{
    Q_OBJECT
public:
    enum CONNECTION_TYPE {
        MAV_TCP = 0,
        MAV_RAGAS = 1,
        MAV_SERIAL = 2,
        MAV_UDP = 3,
    };
    explicit LinkInterfaceManager(QObject *parent = nullptr);
    LinkInterface* linkForAPConnection(CONNECTION_TYPE type);
Q_SIGNALS:

public Q_SLOTS:
};

#endif // LINKINTERFACEMANAGER_H
