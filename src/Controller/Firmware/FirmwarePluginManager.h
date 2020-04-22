#ifndef FIRMWAREPLUGINMANAGER_H
#define FIRMWAREPLUGINMANAGER_H

#include <QObject>
#include "../Com/QGCMAVLink.h"

//#include "FirmwarePlugin.h"
class Vehicle;
class FirmwarePlugin;
class FirmwarePluginManager : public QObject
{
    Q_OBJECT
public:
    explicit FirmwarePluginManager(QObject *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:

public:
    FirmwarePlugin* firmwarePluginForAutopilot(Vehicle*vehicle, MAV_AUTOPILOT firmwareType, MAV_TYPE vehicleType);
};

#endif // FIRMWAREPLUGINMANAGER_H
