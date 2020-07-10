#include "FirmwarePluginManager.h"
#include "APM/QuadPlaneFirmware.h"
#include "APM/ArduCopterFirmware.h"
FirmwarePluginManager::FirmwarePluginManager(QObject *parent) : QObject(parent)
{

}
FirmwarePlugin* FirmwarePluginManager::firmwarePluginForAutopilot(Vehicle* vehicle, MAV_AUTOPILOT autopilotType, MAV_TYPE vehicleType)
{
    printf("Create Firmware[AP:%d VEH:%d]\r\n",autopilotType,vehicleType);
    if (autopilotType == MAV_AUTOPILOT_ARDUPILOTMEGA) {
        switch (vehicleType) {
        case MAV_TYPE_QUADROTOR:
        case MAV_TYPE_OCTOROTOR:
            printf("Create ArduCopterFirmware\r\n");
            return new ArduCopterFirmware(vehicle);
        case MAV_TYPE_HEXAROTOR:
        case MAV_TYPE_TRICOPTER:
        case MAV_TYPE_COAXIAL:
        case MAV_TYPE_HELICOPTER:
        case MAV_TYPE_VTOL_DUOROTOR:
        case MAV_TYPE_VTOL_QUADROTOR:
        case MAV_TYPE_VTOL_TILTROTOR:
        case MAV_TYPE_VTOL_RESERVED2:
        case MAV_TYPE_VTOL_RESERVED3:
        case MAV_TYPE_VTOL_RESERVED4:
        case MAV_TYPE_VTOL_RESERVED5:
        case MAV_TYPE_FIXED_WING:
            printf("Create QuadPlaneFirmware\r\n");
            return new QuadPlaneFirmware(vehicle);
        case MAV_TYPE_GROUND_ROVER:
        case MAV_TYPE_SURFACE_BOAT:
        case MAV_TYPE_SUBMARINE:
        default:
            return new FirmwarePlugin(vehicle);
        }
    }

    return new FirmwarePlugin(vehicle);
}
