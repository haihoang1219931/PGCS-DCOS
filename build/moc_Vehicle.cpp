/****************************************************************************
** Meta object code from reading C++ file 'Vehicle.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Vehicle/Vehicle.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Vehicle.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Vehicle_t {
    QByteArrayData data[320];
    char stringdata0[5124];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Vehicle_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Vehicle_t qt_meta_stringdata_Vehicle = {
    {
QT_MOC_LITERAL(0, 0, 7), // "Vehicle"
QT_MOC_LITERAL(1, 8, 24), // "missingParametersChanged"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 17), // "missingParameters"
QT_MOC_LITERAL(4, 52, 19), // "loadProgressChanged"
QT_MOC_LITERAL(5, 72, 5), // "value"
QT_MOC_LITERAL(6, 78, 22), // "mavlinkMessageReceived"
QT_MOC_LITERAL(7, 101, 17), // "mavlink_message_t"
QT_MOC_LITERAL(8, 119, 7), // "message"
QT_MOC_LITERAL(9, 127, 16), // "mavCommandResult"
QT_MOC_LITERAL(10, 144, 9), // "vehicleId"
QT_MOC_LITERAL(11, 154, 9), // "component"
QT_MOC_LITERAL(12, 164, 7), // "command"
QT_MOC_LITERAL(13, 172, 6), // "result"
QT_MOC_LITERAL(14, 179, 20), // "noReponseFromVehicle"
QT_MOC_LITERAL(15, 200, 19), // "homePositionChanged"
QT_MOC_LITERAL(16, 220, 14), // "QGeoCoordinate"
QT_MOC_LITERAL(17, 235, 19), // "currentHomePosition"
QT_MOC_LITERAL(18, 255, 12), // "armedChanged"
QT_MOC_LITERAL(19, 268, 5), // "armed"
QT_MOC_LITERAL(20, 274, 13), // "landedChanged"
QT_MOC_LITERAL(21, 288, 17), // "flightModeChanged"
QT_MOC_LITERAL(22, 306, 10), // "flightMode"
QT_MOC_LITERAL(23, 317, 18), // "flightModesChanged"
QT_MOC_LITERAL(24, 336, 23), // "flightModesOnAirChanged"
QT_MOC_LITERAL(25, 360, 17), // "coordinateChanged"
QT_MOC_LITERAL(26, 378, 8), // "position"
QT_MOC_LITERAL(27, 387, 27), // "homePositionReceivedChanged"
QT_MOC_LITERAL(28, 415, 23), // "messagesReceivedChanged"
QT_MOC_LITERAL(29, 439, 19), // "messagesSentChanged"
QT_MOC_LITERAL(30, 459, 19), // "messagesLostChanged"
QT_MOC_LITERAL(31, 479, 24), // "remoteControlRSSIChanged"
QT_MOC_LITERAL(32, 504, 7), // "uint8_t"
QT_MOC_LITERAL(33, 512, 4), // "rssi"
QT_MOC_LITERAL(34, 517, 13), // "mavlinkRawImu"
QT_MOC_LITERAL(35, 531, 17), // "mavlinkScaledImu1"
QT_MOC_LITERAL(36, 549, 17), // "mavlinkScaledImu2"
QT_MOC_LITERAL(37, 567, 17), // "mavlinkScaledImu3"
QT_MOC_LITERAL(38, 585, 11), // "rollChanged"
QT_MOC_LITERAL(39, 597, 12), // "pitchChanged"
QT_MOC_LITERAL(40, 610, 14), // "headingChanged"
QT_MOC_LITERAL(41, 625, 15), // "airSpeedChanged"
QT_MOC_LITERAL(42, 641, 23), // "altitudeRelativeChanged"
QT_MOC_LITERAL(43, 665, 21), // "engineSensor_1Changed"
QT_MOC_LITERAL(44, 687, 21), // "engineSensor_2Changed"
QT_MOC_LITERAL(45, 709, 14), // "postionChanged"
QT_MOC_LITERAL(46, 724, 10), // "gpsChanged"
QT_MOC_LITERAL(47, 735, 5), // "valid"
QT_MOC_LITERAL(48, 741, 11), // "linkChanged"
QT_MOC_LITERAL(49, 753, 10), // "ekfChanged"
QT_MOC_LITERAL(50, 764, 11), // "vibeChanged"
QT_MOC_LITERAL(51, 776, 20), // "headingToHomeChanged"
QT_MOC_LITERAL(52, 797, 21), // "distanceToHomeChanged"
QT_MOC_LITERAL(53, 819, 22), // "currentWaypointChanged"
QT_MOC_LITERAL(54, 842, 32), // "distanceToCurrentWaypointChanged"
QT_MOC_LITERAL(55, 875, 21), // "batteryVoltageChanged"
QT_MOC_LITERAL(56, 897, 18), // "batteryAmpeChanged"
QT_MOC_LITERAL(57, 916, 18), // "groundSpeedChanged"
QT_MOC_LITERAL(58, 935, 17), // "climbSpeedChanged"
QT_MOC_LITERAL(59, 953, 19), // "altitudeAMSLChanged"
QT_MOC_LITERAL(60, 973, 18), // "altitudeAGLChanged"
QT_MOC_LITERAL(61, 992, 13), // "latGPSChanged"
QT_MOC_LITERAL(62, 1006, 13), // "lonGPSChanged"
QT_MOC_LITERAL(63, 1020, 14), // "hdopGPSChanged"
QT_MOC_LITERAL(64, 1035, 14), // "vdopGPSChanged"
QT_MOC_LITERAL(65, 1050, 26), // "courseOverGroundGPSChanged"
QT_MOC_LITERAL(66, 1077, 15), // "countGPSChanged"
QT_MOC_LITERAL(67, 1093, 14), // "lockGPSChanged"
QT_MOC_LITERAL(68, 1108, 15), // "fuelUsedChanged"
QT_MOC_LITERAL(69, 1124, 19), // "fuelAvailbleChanged"
QT_MOC_LITERAL(70, 1144, 10), // "uasChanged"
QT_MOC_LITERAL(71, 1155, 22), // "firmwareVersionChanged"
QT_MOC_LITERAL(72, 1178, 28), // "firmwareCustomVersionChanged"
QT_MOC_LITERAL(73, 1207, 17), // "vehicleUIDChanged"
QT_MOC_LITERAL(74, 1225, 22), // "messageSecurityChanged"
QT_MOC_LITERAL(75, 1248, 26), // "_sendMessageOnLinkOnThread"
QT_MOC_LITERAL(76, 1275, 19), // "IOFlightController*"
QT_MOC_LITERAL(77, 1295, 4), // "link"
QT_MOC_LITERAL(78, 1300, 19), // "textMessageReceived"
QT_MOC_LITERAL(79, 1320, 5), // "uasid"
QT_MOC_LITERAL(80, 1326, 11), // "componentid"
QT_MOC_LITERAL(81, 1338, 8), // "severity"
QT_MOC_LITERAL(82, 1347, 4), // "text"
QT_MOC_LITERAL(83, 1352, 23), // "unhealthySensorsChanged"
QT_MOC_LITERAL(84, 1376, 25), // "sensorsPresentBitsChanged"
QT_MOC_LITERAL(85, 1402, 18), // "sensorsPresentBits"
QT_MOC_LITERAL(86, 1421, 25), // "sensorsEnabledBitsChanged"
QT_MOC_LITERAL(87, 1447, 18), // "sensorsEnabledBits"
QT_MOC_LITERAL(88, 1466, 24), // "sensorsHealthBitsChanged"
QT_MOC_LITERAL(89, 1491, 17), // "sensorsHealthBits"
QT_MOC_LITERAL(90, 1509, 20), // "mavlinkStatusChanged"
QT_MOC_LITERAL(91, 1530, 18), // "vehicleTypeChanged"
QT_MOC_LITERAL(92, 1549, 20), // "paramAirSpeedChanged"
QT_MOC_LITERAL(93, 1570, 24), // "paramLoiterRadiusChanged"
QT_MOC_LITERAL(94, 1595, 12), // "paramChanged"
QT_MOC_LITERAL(95, 1608, 4), // "name"
QT_MOC_LITERAL(96, 1613, 11), // "rssiChanged"
QT_MOC_LITERAL(97, 1625, 15), // "pressABSChanged"
QT_MOC_LITERAL(98, 1641, 17), // "sonarRangeChanged"
QT_MOC_LITERAL(99, 1659, 18), // "temperatureChanged"
QT_MOC_LITERAL(100, 1678, 22), // "propertiesModelChanged"
QT_MOC_LITERAL(101, 1701, 26), // "propertiesShowCountChanged"
QT_MOC_LITERAL(102, 1728, 18), // "paramsModelChanged"
QT_MOC_LITERAL(103, 1747, 10), // "picChanged"
QT_MOC_LITERAL(104, 1758, 18), // "useJoystickChanged"
QT_MOC_LITERAL(105, 1777, 6), // "enable"
QT_MOC_LITERAL(106, 1784, 16), // "rcinChan1Changed"
QT_MOC_LITERAL(107, 1801, 16), // "rcinChan2Changed"
QT_MOC_LITERAL(108, 1818, 16), // "rcinChan3Changed"
QT_MOC_LITERAL(109, 1835, 16), // "rcinChan4Changed"
QT_MOC_LITERAL(110, 1852, 22), // "_loadDefaultParamsShow"
QT_MOC_LITERAL(111, 1875, 17), // "_setPropertyValue"
QT_MOC_LITERAL(112, 1893, 4), // "unit"
QT_MOC_LITERAL(113, 1898, 18), // "_sendMessageOnLink"
QT_MOC_LITERAL(114, 1917, 23), // "_mavlinkMessageReceived"
QT_MOC_LITERAL(115, 1941, 17), // "_sendGCSHeartbeat"
QT_MOC_LITERAL(116, 1959, 16), // "_checkCameraLink"
QT_MOC_LITERAL(117, 1976, 14), // "_sendGetParams"
QT_MOC_LITERAL(118, 1991, 21), // "_sendQGCTimeToVehicle"
QT_MOC_LITERAL(119, 2013, 17), // "requestDataStream"
QT_MOC_LITERAL(120, 2031, 9), // "messageID"
QT_MOC_LITERAL(121, 2041, 2), // "hz"
QT_MOC_LITERAL(122, 2044, 17), // "_startPlanRequest"
QT_MOC_LITERAL(123, 2062, 21), // "_mavlinkMessageStatus"
QT_MOC_LITERAL(124, 2084, 5), // "uasId"
QT_MOC_LITERAL(125, 2090, 8), // "uint64_t"
QT_MOC_LITERAL(126, 2099, 9), // "totalSent"
QT_MOC_LITERAL(127, 2109, 13), // "totalReceived"
QT_MOC_LITERAL(128, 2123, 9), // "totalLoss"
QT_MOC_LITERAL(129, 2133, 11), // "lossPercent"
QT_MOC_LITERAL(130, 2145, 9), // "handlePIC"
QT_MOC_LITERAL(131, 2155, 17), // "handleUseJoystick"
QT_MOC_LITERAL(132, 2173, 11), // "useJoystick"
QT_MOC_LITERAL(133, 2185, 19), // "commandLoiterRadius"
QT_MOC_LITERAL(134, 2205, 6), // "radius"
QT_MOC_LITERAL(135, 2212, 10), // "commandRTL"
QT_MOC_LITERAL(136, 2223, 11), // "commandLand"
QT_MOC_LITERAL(137, 2235, 14), // "commandTakeoff"
QT_MOC_LITERAL(138, 2250, 16), // "altitudeRelative"
QT_MOC_LITERAL(139, 2267, 22), // "minimumTakeoffAltitude"
QT_MOC_LITERAL(140, 2290, 19), // "commandGotoLocation"
QT_MOC_LITERAL(141, 2310, 9), // "gotoCoord"
QT_MOC_LITERAL(142, 2320, 21), // "commandChangeAltitude"
QT_MOC_LITERAL(143, 2342, 14), // "altitudeChange"
QT_MOC_LITERAL(144, 2357, 18), // "commandSetAltitude"
QT_MOC_LITERAL(145, 2376, 11), // "newAltitude"
QT_MOC_LITERAL(146, 2388, 18), // "commandChangeSpeed"
QT_MOC_LITERAL(147, 2407, 11), // "speedChange"
QT_MOC_LITERAL(148, 2419, 12), // "commandOrbit"
QT_MOC_LITERAL(149, 2432, 11), // "centerCoord"
QT_MOC_LITERAL(150, 2444, 12), // "amslAltitude"
QT_MOC_LITERAL(151, 2457, 12), // "pauseVehicle"
QT_MOC_LITERAL(152, 2470, 13), // "emergencyStop"
QT_MOC_LITERAL(153, 2484, 12), // "abortLanding"
QT_MOC_LITERAL(154, 2497, 16), // "climbOutAltitude"
QT_MOC_LITERAL(155, 2514, 12), // "startMission"
QT_MOC_LITERAL(156, 2527, 11), // "startEngine"
QT_MOC_LITERAL(157, 2539, 25), // "setCurrentMissionSequence"
QT_MOC_LITERAL(158, 2565, 3), // "seq"
QT_MOC_LITERAL(159, 2569, 13), // "rebootVehicle"
QT_MOC_LITERAL(160, 2583, 13), // "clearMessages"
QT_MOC_LITERAL(161, 2597, 13), // "triggerCamera"
QT_MOC_LITERAL(162, 2611, 8), // "sendPlan"
QT_MOC_LITERAL(163, 2620, 8), // "planFile"
QT_MOC_LITERAL(164, 2629, 14), // "versionCompare"
QT_MOC_LITERAL(165, 2644, 8), // "QString&"
QT_MOC_LITERAL(166, 2653, 7), // "compare"
QT_MOC_LITERAL(167, 2661, 5), // "major"
QT_MOC_LITERAL(168, 2667, 5), // "minor"
QT_MOC_LITERAL(169, 2673, 5), // "patch"
QT_MOC_LITERAL(170, 2679, 9), // "motorTest"
QT_MOC_LITERAL(171, 2689, 5), // "motor"
QT_MOC_LITERAL(172, 2695, 7), // "percent"
QT_MOC_LITERAL(173, 2703, 15), // "setHomeLocation"
QT_MOC_LITERAL(174, 2719, 3), // "lat"
QT_MOC_LITERAL(175, 2723, 3), // "lon"
QT_MOC_LITERAL(176, 2727, 14), // "setAltitudeRTL"
QT_MOC_LITERAL(177, 2742, 3), // "alt"
QT_MOC_LITERAL(178, 2746, 16), // "sendHomePosition"
QT_MOC_LITERAL(179, 2763, 8), // "location"
QT_MOC_LITERAL(180, 2772, 14), // "activeProperty"
QT_MOC_LITERAL(181, 2787, 6), // "active"
QT_MOC_LITERAL(182, 2794, 21), // "countActiveProperties"
QT_MOC_LITERAL(183, 2816, 8), // "setArmed"
QT_MOC_LITERAL(184, 2825, 11), // "sendCommand"
QT_MOC_LITERAL(185, 2837, 9), // "showError"
QT_MOC_LITERAL(186, 2847, 6), // "param1"
QT_MOC_LITERAL(187, 2854, 6), // "param2"
QT_MOC_LITERAL(188, 2861, 6), // "param3"
QT_MOC_LITERAL(189, 2868, 6), // "param4"
QT_MOC_LITERAL(190, 2875, 6), // "param5"
QT_MOC_LITERAL(191, 2882, 6), // "param6"
QT_MOC_LITERAL(192, 2889, 6), // "param7"
QT_MOC_LITERAL(193, 2896, 3), // "uav"
QT_MOC_LITERAL(194, 2900, 8), // "Vehicle*"
QT_MOC_LITERAL(195, 2909, 8), // "joystick"
QT_MOC_LITERAL(196, 2918, 17), // "JoystickThreaded*"
QT_MOC_LITERAL(197, 2936, 13), // "communication"
QT_MOC_LITERAL(198, 2950, 14), // "planController"
QT_MOC_LITERAL(199, 2965, 15), // "PlanController*"
QT_MOC_LITERAL(200, 2981, 16), // "paramsController"
QT_MOC_LITERAL(201, 2998, 17), // "ParamsController*"
QT_MOC_LITERAL(202, 3016, 11), // "flightModes"
QT_MOC_LITERAL(203, 3028, 16), // "flightModesOnAir"
QT_MOC_LITERAL(204, 3045, 9), // "rcinChan1"
QT_MOC_LITERAL(205, 3055, 9), // "rcinChan2"
QT_MOC_LITERAL(206, 3065, 9), // "rcinChan3"
QT_MOC_LITERAL(207, 3075, 9), // "rcinChan4"
QT_MOC_LITERAL(208, 3085, 3), // "pic"
QT_MOC_LITERAL(209, 3089, 6), // "landed"
QT_MOC_LITERAL(210, 3096, 10), // "coordinate"
QT_MOC_LITERAL(211, 3107, 12), // "homePosition"
QT_MOC_LITERAL(212, 3120, 4), // "roll"
QT_MOC_LITERAL(213, 3125, 5), // "pitch"
QT_MOC_LITERAL(214, 3131, 7), // "heading"
QT_MOC_LITERAL(215, 3139, 8), // "airSpeed"
QT_MOC_LITERAL(216, 3148, 14), // "engineSensor_1"
QT_MOC_LITERAL(217, 3163, 14), // "engineSensor_2"
QT_MOC_LITERAL(218, 3178, 9), // "gpsSignal"
QT_MOC_LITERAL(219, 3188, 9), // "ekfSignal"
QT_MOC_LITERAL(220, 3198, 10), // "vibeSignal"
QT_MOC_LITERAL(221, 3209, 13), // "headingToHome"
QT_MOC_LITERAL(222, 3223, 14), // "distanceToHome"
QT_MOC_LITERAL(223, 3238, 15), // "currentWaypoint"
QT_MOC_LITERAL(224, 3254, 25), // "distanceToCurrentWaypoint"
QT_MOC_LITERAL(225, 3280, 14), // "batteryVoltage"
QT_MOC_LITERAL(226, 3295, 11), // "batteryAmpe"
QT_MOC_LITERAL(227, 3307, 11), // "groundSpeed"
QT_MOC_LITERAL(228, 3319, 10), // "climbSpeed"
QT_MOC_LITERAL(229, 3330, 12), // "altitudeAMSL"
QT_MOC_LITERAL(230, 3343, 11), // "altitudeAGL"
QT_MOC_LITERAL(231, 3355, 6), // "latGPS"
QT_MOC_LITERAL(232, 3362, 6), // "lonGPS"
QT_MOC_LITERAL(233, 3369, 7), // "hdopGPS"
QT_MOC_LITERAL(234, 3377, 7), // "vdopGPS"
QT_MOC_LITERAL(235, 3385, 19), // "courseOverGroundGPS"
QT_MOC_LITERAL(236, 3405, 8), // "countGPS"
QT_MOC_LITERAL(237, 3414, 7), // "lockGPS"
QT_MOC_LITERAL(238, 3422, 8), // "fuelUsed"
QT_MOC_LITERAL(239, 3431, 12), // "fuelAvailble"
QT_MOC_LITERAL(240, 3444, 15), // "messageSecurity"
QT_MOC_LITERAL(241, 3460, 3), // "uas"
QT_MOC_LITERAL(242, 3464, 4), // "UAS*"
QT_MOC_LITERAL(243, 3469, 16), // "unhealthySensors"
QT_MOC_LITERAL(244, 3486, 16), // "mavlinkSentCount"
QT_MOC_LITERAL(245, 3503, 20), // "mavlinkReceivedCount"
QT_MOC_LITERAL(246, 3524, 16), // "mavlinkLossCount"
QT_MOC_LITERAL(247, 3541, 18), // "mavlinkLossPercent"
QT_MOC_LITERAL(248, 3560, 11), // "vehicleType"
QT_MOC_LITERAL(249, 3572, 16), // "VEHICLE_MAV_TYPE"
QT_MOC_LITERAL(250, 3589, 13), // "paramAirSpeed"
QT_MOC_LITERAL(251, 3603, 17), // "paramLoiterRadius"
QT_MOC_LITERAL(252, 3621, 8), // "pressABS"
QT_MOC_LITERAL(253, 3630, 10), // "sonarRange"
QT_MOC_LITERAL(254, 3641, 11), // "temperature"
QT_MOC_LITERAL(255, 3653, 15), // "propertiesModel"
QT_MOC_LITERAL(256, 3669, 22), // "QQmlListProperty<Fact>"
QT_MOC_LITERAL(257, 3692, 19), // "propertiesShowCount"
QT_MOC_LITERAL(258, 3712, 11), // "paramsModel"
QT_MOC_LITERAL(259, 3724, 16), // "MavlinkSysStatus"
QT_MOC_LITERAL(260, 3741, 21), // "SysStatusSensor3dGyro"
QT_MOC_LITERAL(261, 3763, 22), // "SysStatusSensor3dAccel"
QT_MOC_LITERAL(262, 3786, 20), // "SysStatusSensor3dMag"
QT_MOC_LITERAL(263, 3807, 31), // "SysStatusSensorAbsolutePressure"
QT_MOC_LITERAL(264, 3839, 35), // "SysStatusSensorDifferentialPr..."
QT_MOC_LITERAL(265, 3875, 18), // "SysStatusSensorGPS"
QT_MOC_LITERAL(266, 3894, 26), // "SysStatusSensorOpticalFlow"
QT_MOC_LITERAL(267, 3921, 29), // "SysStatusSensorVisionPosition"
QT_MOC_LITERAL(268, 3951, 28), // "SysStatusSensorLaserPosition"
QT_MOC_LITERAL(269, 3980, 34), // "SysStatusSensorExternalGround..."
QT_MOC_LITERAL(270, 4015, 33), // "SysStatusSensorAngularRateCon..."
QT_MOC_LITERAL(271, 4049, 36), // "SysStatusSensorAttitudeStabil..."
QT_MOC_LITERAL(272, 4086, 26), // "SysStatusSensorYawPosition"
QT_MOC_LITERAL(273, 4113, 31), // "SysStatusSensorZAltitudeControl"
QT_MOC_LITERAL(274, 4145, 32), // "SysStatusSensorXYPositionControl"
QT_MOC_LITERAL(275, 4178, 27), // "SysStatusSensorMotorOutputs"
QT_MOC_LITERAL(276, 4206, 25), // "SysStatusSensorRCReceiver"
QT_MOC_LITERAL(277, 4232, 22), // "SysStatusSensor3dGyro2"
QT_MOC_LITERAL(278, 4255, 23), // "SysStatusSensor3dAccel2"
QT_MOC_LITERAL(279, 4279, 21), // "SysStatusSensor3dMag2"
QT_MOC_LITERAL(280, 4301, 23), // "SysStatusSensorGeoFence"
QT_MOC_LITERAL(281, 4325, 19), // "SysStatusSensorAHRS"
QT_MOC_LITERAL(282, 4345, 22), // "SysStatusSensorTerrain"
QT_MOC_LITERAL(283, 4368, 27), // "SysStatusSensorReverseMotor"
QT_MOC_LITERAL(284, 4396, 22), // "SysStatusSensorLogging"
QT_MOC_LITERAL(285, 4419, 22), // "SysStatusSensorBattery"
QT_MOC_LITERAL(286, 4442, 16), // "MAV_TYPE_GENERIC"
QT_MOC_LITERAL(287, 4459, 19), // "MAV_TYPE_FIXED_WING"
QT_MOC_LITERAL(288, 4479, 18), // "MAV_TYPE_QUADROTOR"
QT_MOC_LITERAL(289, 4498, 16), // "MAV_TYPE_COAXIAL"
QT_MOC_LITERAL(290, 4515, 19), // "MAV_TYPE_HELICOPTER"
QT_MOC_LITERAL(291, 4535, 24), // "MAV_TYPE_ANTENNA_TRACKER"
QT_MOC_LITERAL(292, 4560, 12), // "MAV_TYPE_GCS"
QT_MOC_LITERAL(293, 4573, 16), // "MAV_TYPE_AIRSHIP"
QT_MOC_LITERAL(294, 4590, 21), // "MAV_TYPE_FREE_BALLOON"
QT_MOC_LITERAL(295, 4612, 15), // "MAV_TYPE_ROCKET"
QT_MOC_LITERAL(296, 4628, 21), // "MAV_TYPE_GROUND_ROVER"
QT_MOC_LITERAL(297, 4650, 21), // "MAV_TYPE_SURFACE_BOAT"
QT_MOC_LITERAL(298, 4672, 18), // "MAV_TYPE_SUBMARINE"
QT_MOC_LITERAL(299, 4691, 18), // "MAV_TYPE_HEXAROTOR"
QT_MOC_LITERAL(300, 4710, 18), // "MAV_TYPE_OCTOROTOR"
QT_MOC_LITERAL(301, 4729, 18), // "MAV_TYPE_TRICOPTER"
QT_MOC_LITERAL(302, 4748, 22), // "MAV_TYPE_FLAPPING_WING"
QT_MOC_LITERAL(303, 4771, 13), // "MAV_TYPE_KITE"
QT_MOC_LITERAL(304, 4785, 27), // "MAV_TYPE_ONBOARD_CONTROLLER"
QT_MOC_LITERAL(305, 4813, 22), // "MAV_TYPE_VTOL_DUOROTOR"
QT_MOC_LITERAL(306, 4836, 23), // "MAV_TYPE_VTOL_QUADROTOR"
QT_MOC_LITERAL(307, 4860, 23), // "MAV_TYPE_VTOL_TILTROTOR"
QT_MOC_LITERAL(308, 4884, 23), // "MAV_TYPE_VTOL_RESERVED2"
QT_MOC_LITERAL(309, 4908, 23), // "MAV_TYPE_VTOL_RESERVED3"
QT_MOC_LITERAL(310, 4932, 23), // "MAV_TYPE_VTOL_RESERVED4"
QT_MOC_LITERAL(311, 4956, 23), // "MAV_TYPE_VTOL_RESERVED5"
QT_MOC_LITERAL(312, 4980, 15), // "MAV_TYPE_GIMBAL"
QT_MOC_LITERAL(313, 4996, 13), // "MAV_TYPE_ADSB"
QT_MOC_LITERAL(314, 5010, 17), // "MAV_TYPE_PARAFOIL"
QT_MOC_LITERAL(315, 5028, 20), // "MAV_TYPE_DODECAROTOR"
QT_MOC_LITERAL(316, 5049, 15), // "MAV_TYPE_CAMERA"
QT_MOC_LITERAL(317, 5065, 25), // "MAV_TYPE_CHARGING_STATION"
QT_MOC_LITERAL(318, 5091, 14), // "MAV_TYPE_FLARM"
QT_MOC_LITERAL(319, 5106, 17) // "MAV_TYPE_ENUM_END"

    },
    "Vehicle\0missingParametersChanged\0\0"
    "missingParameters\0loadProgressChanged\0"
    "value\0mavlinkMessageReceived\0"
    "mavlink_message_t\0message\0mavCommandResult\0"
    "vehicleId\0component\0command\0result\0"
    "noReponseFromVehicle\0homePositionChanged\0"
    "QGeoCoordinate\0currentHomePosition\0"
    "armedChanged\0armed\0landedChanged\0"
    "flightModeChanged\0flightMode\0"
    "flightModesChanged\0flightModesOnAirChanged\0"
    "coordinateChanged\0position\0"
    "homePositionReceivedChanged\0"
    "messagesReceivedChanged\0messagesSentChanged\0"
    "messagesLostChanged\0remoteControlRSSIChanged\0"
    "uint8_t\0rssi\0mavlinkRawImu\0mavlinkScaledImu1\0"
    "mavlinkScaledImu2\0mavlinkScaledImu3\0"
    "rollChanged\0pitchChanged\0headingChanged\0"
    "airSpeedChanged\0altitudeRelativeChanged\0"
    "engineSensor_1Changed\0engineSensor_2Changed\0"
    "postionChanged\0gpsChanged\0valid\0"
    "linkChanged\0ekfChanged\0vibeChanged\0"
    "headingToHomeChanged\0distanceToHomeChanged\0"
    "currentWaypointChanged\0"
    "distanceToCurrentWaypointChanged\0"
    "batteryVoltageChanged\0batteryAmpeChanged\0"
    "groundSpeedChanged\0climbSpeedChanged\0"
    "altitudeAMSLChanged\0altitudeAGLChanged\0"
    "latGPSChanged\0lonGPSChanged\0hdopGPSChanged\0"
    "vdopGPSChanged\0courseOverGroundGPSChanged\0"
    "countGPSChanged\0lockGPSChanged\0"
    "fuelUsedChanged\0fuelAvailbleChanged\0"
    "uasChanged\0firmwareVersionChanged\0"
    "firmwareCustomVersionChanged\0"
    "vehicleUIDChanged\0messageSecurityChanged\0"
    "_sendMessageOnLinkOnThread\0"
    "IOFlightController*\0link\0textMessageReceived\0"
    "uasid\0componentid\0severity\0text\0"
    "unhealthySensorsChanged\0"
    "sensorsPresentBitsChanged\0sensorsPresentBits\0"
    "sensorsEnabledBitsChanged\0sensorsEnabledBits\0"
    "sensorsHealthBitsChanged\0sensorsHealthBits\0"
    "mavlinkStatusChanged\0vehicleTypeChanged\0"
    "paramAirSpeedChanged\0paramLoiterRadiusChanged\0"
    "paramChanged\0name\0rssiChanged\0"
    "pressABSChanged\0sonarRangeChanged\0"
    "temperatureChanged\0propertiesModelChanged\0"
    "propertiesShowCountChanged\0"
    "paramsModelChanged\0picChanged\0"
    "useJoystickChanged\0enable\0rcinChan1Changed\0"
    "rcinChan2Changed\0rcinChan3Changed\0"
    "rcinChan4Changed\0_loadDefaultParamsShow\0"
    "_setPropertyValue\0unit\0_sendMessageOnLink\0"
    "_mavlinkMessageReceived\0_sendGCSHeartbeat\0"
    "_checkCameraLink\0_sendGetParams\0"
    "_sendQGCTimeToVehicle\0requestDataStream\0"
    "messageID\0hz\0_startPlanRequest\0"
    "_mavlinkMessageStatus\0uasId\0uint64_t\0"
    "totalSent\0totalReceived\0totalLoss\0"
    "lossPercent\0handlePIC\0handleUseJoystick\0"
    "useJoystick\0commandLoiterRadius\0radius\0"
    "commandRTL\0commandLand\0commandTakeoff\0"
    "altitudeRelative\0minimumTakeoffAltitude\0"
    "commandGotoLocation\0gotoCoord\0"
    "commandChangeAltitude\0altitudeChange\0"
    "commandSetAltitude\0newAltitude\0"
    "commandChangeSpeed\0speedChange\0"
    "commandOrbit\0centerCoord\0amslAltitude\0"
    "pauseVehicle\0emergencyStop\0abortLanding\0"
    "climbOutAltitude\0startMission\0startEngine\0"
    "setCurrentMissionSequence\0seq\0"
    "rebootVehicle\0clearMessages\0triggerCamera\0"
    "sendPlan\0planFile\0versionCompare\0"
    "QString&\0compare\0major\0minor\0patch\0"
    "motorTest\0motor\0percent\0setHomeLocation\0"
    "lat\0lon\0setAltitudeRTL\0alt\0sendHomePosition\0"
    "location\0activeProperty\0active\0"
    "countActiveProperties\0setArmed\0"
    "sendCommand\0showError\0param1\0param2\0"
    "param3\0param4\0param5\0param6\0param7\0"
    "uav\0Vehicle*\0joystick\0JoystickThreaded*\0"
    "communication\0planController\0"
    "PlanController*\0paramsController\0"
    "ParamsController*\0flightModes\0"
    "flightModesOnAir\0rcinChan1\0rcinChan2\0"
    "rcinChan3\0rcinChan4\0pic\0landed\0"
    "coordinate\0homePosition\0roll\0pitch\0"
    "heading\0airSpeed\0engineSensor_1\0"
    "engineSensor_2\0gpsSignal\0ekfSignal\0"
    "vibeSignal\0headingToHome\0distanceToHome\0"
    "currentWaypoint\0distanceToCurrentWaypoint\0"
    "batteryVoltage\0batteryAmpe\0groundSpeed\0"
    "climbSpeed\0altitudeAMSL\0altitudeAGL\0"
    "latGPS\0lonGPS\0hdopGPS\0vdopGPS\0"
    "courseOverGroundGPS\0countGPS\0lockGPS\0"
    "fuelUsed\0fuelAvailble\0messageSecurity\0"
    "uas\0UAS*\0unhealthySensors\0mavlinkSentCount\0"
    "mavlinkReceivedCount\0mavlinkLossCount\0"
    "mavlinkLossPercent\0vehicleType\0"
    "VEHICLE_MAV_TYPE\0paramAirSpeed\0"
    "paramLoiterRadius\0pressABS\0sonarRange\0"
    "temperature\0propertiesModel\0"
    "QQmlListProperty<Fact>\0propertiesShowCount\0"
    "paramsModel\0MavlinkSysStatus\0"
    "SysStatusSensor3dGyro\0SysStatusSensor3dAccel\0"
    "SysStatusSensor3dMag\0"
    "SysStatusSensorAbsolutePressure\0"
    "SysStatusSensorDifferentialPressure\0"
    "SysStatusSensorGPS\0SysStatusSensorOpticalFlow\0"
    "SysStatusSensorVisionPosition\0"
    "SysStatusSensorLaserPosition\0"
    "SysStatusSensorExternalGroundTruth\0"
    "SysStatusSensorAngularRateControl\0"
    "SysStatusSensorAttitudeStabilization\0"
    "SysStatusSensorYawPosition\0"
    "SysStatusSensorZAltitudeControl\0"
    "SysStatusSensorXYPositionControl\0"
    "SysStatusSensorMotorOutputs\0"
    "SysStatusSensorRCReceiver\0"
    "SysStatusSensor3dGyro2\0SysStatusSensor3dAccel2\0"
    "SysStatusSensor3dMag2\0SysStatusSensorGeoFence\0"
    "SysStatusSensorAHRS\0SysStatusSensorTerrain\0"
    "SysStatusSensorReverseMotor\0"
    "SysStatusSensorLogging\0SysStatusSensorBattery\0"
    "MAV_TYPE_GENERIC\0MAV_TYPE_FIXED_WING\0"
    "MAV_TYPE_QUADROTOR\0MAV_TYPE_COAXIAL\0"
    "MAV_TYPE_HELICOPTER\0MAV_TYPE_ANTENNA_TRACKER\0"
    "MAV_TYPE_GCS\0MAV_TYPE_AIRSHIP\0"
    "MAV_TYPE_FREE_BALLOON\0MAV_TYPE_ROCKET\0"
    "MAV_TYPE_GROUND_ROVER\0MAV_TYPE_SURFACE_BOAT\0"
    "MAV_TYPE_SUBMARINE\0MAV_TYPE_HEXAROTOR\0"
    "MAV_TYPE_OCTOROTOR\0MAV_TYPE_TRICOPTER\0"
    "MAV_TYPE_FLAPPING_WING\0MAV_TYPE_KITE\0"
    "MAV_TYPE_ONBOARD_CONTROLLER\0"
    "MAV_TYPE_VTOL_DUOROTOR\0MAV_TYPE_VTOL_QUADROTOR\0"
    "MAV_TYPE_VTOL_TILTROTOR\0MAV_TYPE_VTOL_RESERVED2\0"
    "MAV_TYPE_VTOL_RESERVED3\0MAV_TYPE_VTOL_RESERVED4\0"
    "MAV_TYPE_VTOL_RESERVED5\0MAV_TYPE_GIMBAL\0"
    "MAV_TYPE_ADSB\0MAV_TYPE_PARAFOIL\0"
    "MAV_TYPE_DODECAROTOR\0MAV_TYPE_CAMERA\0"
    "MAV_TYPE_CHARGING_STATION\0MAV_TYPE_FLARM\0"
    "MAV_TYPE_ENUM_END"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Vehicle[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
     131,   14, // methods
      68, 1046, // properties
       2, 1318, // enums/sets
       0,    0, // constructors
       0,       // flags
      80,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  669,    2, 0x06 /* Public */,
       4,    1,  672,    2, 0x06 /* Public */,
       6,    1,  675,    2, 0x06 /* Public */,
       9,    5,  678,    2, 0x06 /* Public */,
      15,    1,  689,    2, 0x06 /* Public */,
      18,    1,  692,    2, 0x06 /* Public */,
      20,    0,  695,    2, 0x06 /* Public */,
      21,    1,  696,    2, 0x06 /* Public */,
      23,    0,  699,    2, 0x06 /* Public */,
      24,    0,  700,    2, 0x06 /* Public */,
      25,    1,  701,    2, 0x06 /* Public */,
      27,    0,  704,    2, 0x06 /* Public */,
      28,    0,  705,    2, 0x06 /* Public */,
      29,    0,  706,    2, 0x06 /* Public */,
      30,    0,  707,    2, 0x06 /* Public */,
      31,    1,  708,    2, 0x06 /* Public */,
      34,    1,  711,    2, 0x06 /* Public */,
      35,    1,  714,    2, 0x06 /* Public */,
      36,    1,  717,    2, 0x06 /* Public */,
      37,    1,  720,    2, 0x06 /* Public */,
      38,    0,  723,    2, 0x06 /* Public */,
      39,    0,  724,    2, 0x06 /* Public */,
      40,    0,  725,    2, 0x06 /* Public */,
      41,    0,  726,    2, 0x06 /* Public */,
      42,    0,  727,    2, 0x06 /* Public */,
      43,    0,  728,    2, 0x06 /* Public */,
      44,    0,  729,    2, 0x06 /* Public */,
      45,    0,  730,    2, 0x06 /* Public */,
      46,    1,  731,    2, 0x06 /* Public */,
      48,    0,  734,    2, 0x06 /* Public */,
      49,    0,  735,    2, 0x06 /* Public */,
      50,    0,  736,    2, 0x06 /* Public */,
      51,    0,  737,    2, 0x06 /* Public */,
      52,    0,  738,    2, 0x06 /* Public */,
      53,    0,  739,    2, 0x06 /* Public */,
      54,    0,  740,    2, 0x06 /* Public */,
      55,    0,  741,    2, 0x06 /* Public */,
      56,    0,  742,    2, 0x06 /* Public */,
      57,    0,  743,    2, 0x06 /* Public */,
      58,    0,  744,    2, 0x06 /* Public */,
      59,    0,  745,    2, 0x06 /* Public */,
      60,    0,  746,    2, 0x06 /* Public */,
      61,    0,  747,    2, 0x06 /* Public */,
      62,    0,  748,    2, 0x06 /* Public */,
      63,    0,  749,    2, 0x06 /* Public */,
      64,    0,  750,    2, 0x06 /* Public */,
      65,    0,  751,    2, 0x06 /* Public */,
      66,    0,  752,    2, 0x06 /* Public */,
      67,    0,  753,    2, 0x06 /* Public */,
      68,    0,  754,    2, 0x06 /* Public */,
      69,    0,  755,    2, 0x06 /* Public */,
      70,    0,  756,    2, 0x06 /* Public */,
      71,    0,  757,    2, 0x06 /* Public */,
      72,    0,  758,    2, 0x06 /* Public */,
      73,    0,  759,    2, 0x06 /* Public */,
      74,    0,  760,    2, 0x06 /* Public */,
      75,    2,  761,    2, 0x06 /* Public */,
      78,    4,  766,    2, 0x06 /* Public */,
      83,    0,  775,    2, 0x06 /* Public */,
      84,    1,  776,    2, 0x06 /* Public */,
      86,    1,  779,    2, 0x06 /* Public */,
      88,    1,  782,    2, 0x06 /* Public */,
      90,    0,  785,    2, 0x06 /* Public */,
      91,    0,  786,    2, 0x06 /* Public */,
      92,    0,  787,    2, 0x06 /* Public */,
      93,    0,  788,    2, 0x06 /* Public */,
      94,    1,  789,    2, 0x06 /* Public */,
      96,    0,  792,    2, 0x06 /* Public */,
      97,    0,  793,    2, 0x06 /* Public */,
      98,    0,  794,    2, 0x06 /* Public */,
      99,    0,  795,    2, 0x06 /* Public */,
     100,    0,  796,    2, 0x06 /* Public */,
     101,    0,  797,    2, 0x06 /* Public */,
     102,    0,  798,    2, 0x06 /* Public */,
     103,    0,  799,    2, 0x06 /* Public */,
     104,    1,  800,    2, 0x06 /* Public */,
     106,    0,  803,    2, 0x06 /* Public */,
     107,    0,  804,    2, 0x06 /* Public */,
     108,    0,  805,    2, 0x06 /* Public */,
     109,    0,  806,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
     110,    0,  807,    2, 0x0a /* Public */,
     111,    3,  808,    2, 0x0a /* Public */,
     113,    2,  815,    2, 0x0a /* Public */,
     114,    1,  820,    2, 0x0a /* Public */,
     115,    0,  823,    2, 0x0a /* Public */,
     116,    0,  824,    2, 0x0a /* Public */,
     117,    0,  825,    2, 0x0a /* Public */,
     118,    0,  826,    2, 0x0a /* Public */,
     119,    3,  827,    2, 0x0a /* Public */,
     119,    2,  834,    2, 0x2a /* Public | MethodCloned */,
     122,    0,  839,    2, 0x0a /* Public */,
     123,    5,  840,    2, 0x0a /* Public */,
     130,    0,  851,    2, 0x0a /* Public */,
     131,    1,  852,    2, 0x0a /* Public */,

 // methods: name, argc, parameters, tag, flags
     133,    1,  855,    2, 0x02 /* Public */,
     135,    0,  858,    2, 0x02 /* Public */,
     136,    0,  859,    2, 0x02 /* Public */,
     137,    1,  860,    2, 0x02 /* Public */,
     139,    0,  863,    2, 0x02 /* Public */,
     140,    1,  864,    2, 0x02 /* Public */,
     142,    1,  867,    2, 0x02 /* Public */,
     144,    1,  870,    2, 0x02 /* Public */,
     146,    1,  873,    2, 0x02 /* Public */,
     148,    3,  876,    2, 0x02 /* Public */,
     151,    0,  883,    2, 0x02 /* Public */,
     152,    0,  884,    2, 0x02 /* Public */,
     153,    1,  885,    2, 0x02 /* Public */,
     155,    0,  888,    2, 0x02 /* Public */,
     156,    0,  889,    2, 0x02 /* Public */,
     157,    1,  890,    2, 0x02 /* Public */,
     159,    0,  893,    2, 0x02 /* Public */,
     160,    0,  894,    2, 0x02 /* Public */,
     161,    0,  895,    2, 0x02 /* Public */,
     162,    1,  896,    2, 0x02 /* Public */,
     164,    1,  899,    2, 0x02 /* Public */,
     164,    3,  902,    2, 0x02 /* Public */,
     170,    2,  909,    2, 0x02 /* Public */,
     173,    2,  914,    2, 0x02 /* Public */,
     176,    1,  919,    2, 0x02 /* Public */,
     178,    1,  922,    2, 0x02 /* Public */,
     180,    2,  925,    2, 0x02 /* Public */,
     182,    0,  930,    2, 0x02 /* Public */,
     183,    1,  931,    2, 0x02 /* Public */,
     184,   10,  934,    2, 0x02 /* Public */,
     184,    9,  955,    2, 0x22 /* Public | MethodCloned */,
     184,    8,  974,    2, 0x22 /* Public | MethodCloned */,
     184,    7,  991,    2, 0x22 /* Public | MethodCloned */,
     184,    6, 1006,    2, 0x22 /* Public | MethodCloned */,
     184,    5, 1019,    2, 0x22 /* Public | MethodCloned */,
     184,    4, 1030,    2, 0x22 /* Public | MethodCloned */,
     184,    3, 1039,    2, 0x22 /* Public | MethodCloned */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Float,    5,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Int, QMetaType::Int, QMetaType::Bool,   10,   11,   12,   13,   14,
    QMetaType::Void, 0x80000000 | 16,   17,
    QMetaType::Void, QMetaType::Bool,   19,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   22,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 16,   26,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 32,   33,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   47,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 76, 0x80000000 | 7,   77,    8,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Int, QMetaType::QString,   79,   80,   81,   82,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   85,
    QMetaType::Void, QMetaType::Int,   87,
    QMetaType::Void, QMetaType::Int,   89,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   95,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,  105,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString,   95,    5,  112,
    QMetaType::Void, 0x80000000 | 76, 0x80000000 | 7,   77,    8,
    QMetaType::Void, 0x80000000 | 7,    8,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Int,  120,  121,  105,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,  120,  121,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, 0x80000000 | 125, 0x80000000 | 125, 0x80000000 | 125, QMetaType::Float,  124,  126,  127,  128,  129,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,  132,

 // methods: parameters
    QMetaType::Void, QMetaType::Float,  134,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,  138,
    QMetaType::Double,
    QMetaType::Void, 0x80000000 | 16,  141,
    QMetaType::Void, QMetaType::Double,  143,
    QMetaType::Void, QMetaType::Double,  145,
    QMetaType::Void, QMetaType::Double,  147,
    QMetaType::Void, 0x80000000 | 16, QMetaType::Double, QMetaType::Double,  149,  134,  150,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,  154,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,  158,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,  163,
    QMetaType::Int, 0x80000000 | 165,  166,
    QMetaType::Int, QMetaType::Int, QMetaType::Int, QMetaType::Int,  167,  168,  169,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,  171,  172,
    QMetaType::Void, QMetaType::Float, QMetaType::Float,  174,  175,
    QMetaType::Void, QMetaType::Float,  177,
    QMetaType::Void, 0x80000000 | 16,  179,
    QMetaType::Void, QMetaType::QString, QMetaType::Bool,   95,  181,
    QMetaType::Int,
    QMetaType::Void, QMetaType::Bool,   19,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,   11,   12,  185,  186,  187,  188,  189,  190,  191,  192,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,   11,   12,  185,  186,  187,  188,  189,  190,  191,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,   11,   12,  185,  186,  187,  188,  189,  190,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,   11,   12,  185,  186,  187,  188,  189,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double, QMetaType::Double, QMetaType::Double,   11,   12,  185,  186,  187,  188,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double, QMetaType::Double,   11,   12,  185,  186,  187,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool, QMetaType::Double,   11,   12,  185,  186,
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Bool,   11,   12,  185,

 // properties: name, type, flags
     193, 0x80000000 | 194, 0x0009510b,
     195, 0x80000000 | 196, 0x0009510b,
     197, 0x80000000 | 76, 0x0009510b,
     198, 0x80000000 | 199, 0x0009510b,
     200, 0x80000000 | 201, 0x0009510b,
     202, QMetaType::QStringList, 0x00495001,
     203, QMetaType::QStringList, 0x00495001,
      22, QMetaType::QString, 0x00495103,
     204, QMetaType::Float, 0x00495001,
     205, QMetaType::Float, 0x00495001,
     206, QMetaType::Float, 0x00495001,
     207, QMetaType::Float, 0x00495001,
     132, QMetaType::Bool, 0x00495103,
     208, QMetaType::Bool, 0x00495001,
      19, QMetaType::Bool, 0x00495001,
     209, QMetaType::Bool, 0x00495001,
     210, 0x80000000 | 16, 0x00495009,
     211, 0x80000000 | 16, 0x0049510b,
     212, QMetaType::Float, 0x00495001,
     213, QMetaType::Float, 0x00495001,
     214, QMetaType::Float, 0x00495001,
     215, QMetaType::Float, 0x00495001,
     138, QMetaType::Float, 0x00495001,
     216, QMetaType::Float, 0x00495001,
     217, QMetaType::Float, 0x00495001,
     218, QMetaType::Bool, 0x00495001,
      77, QMetaType::Bool, 0x00495103,
     219, QMetaType::QString, 0x00495103,
     220, QMetaType::QString, 0x00495103,
     221, QMetaType::Float, 0x00495001,
     222, QMetaType::Float, 0x00495001,
     223, QMetaType::Int, 0x00495001,
     224, QMetaType::Float, 0x00495001,
     225, QMetaType::Float, 0x00495001,
     226, QMetaType::Float, 0x00495001,
     227, QMetaType::Float, 0x00495001,
     228, QMetaType::Float, 0x00495001,
     229, QMetaType::Float, 0x00495001,
     230, QMetaType::Float, 0x00495001,
     231, QMetaType::Float, 0x00495001,
     232, QMetaType::Float, 0x00495001,
     233, QMetaType::Float, 0x00495001,
     234, QMetaType::Float, 0x00495001,
     235, QMetaType::Float, 0x00495001,
     236, QMetaType::Int, 0x00495001,
     237, QMetaType::QString, 0x00495001,
     238, QMetaType::Float, 0x00495001,
     239, QMetaType::Float, 0x00495001,
     240, QMetaType::QString, 0x00495103,
     241, 0x80000000 | 242, 0x00495009,
     243, QMetaType::QStringList, 0x00495001,
      85, QMetaType::Int, 0x00495001,
      87, QMetaType::Int, 0x00495001,
      89, QMetaType::Int, 0x00495001,
     244, QMetaType::ULongLong, 0x00495001,
     245, QMetaType::ULongLong, 0x00495001,
     246, QMetaType::ULongLong, 0x00495001,
     247, QMetaType::Float, 0x00495001,
     248, 0x80000000 | 249, 0x0049510b,
     250, QMetaType::Float, 0x00495103,
     251, QMetaType::Float, 0x00495103,
      33, QMetaType::Int, 0x00495001,
     252, QMetaType::Float, 0x00495001,
     253, QMetaType::Float, 0x00495001,
     254, QMetaType::Int, 0x00495001,
     255, 0x80000000 | 256, 0x00495009,
     257, QMetaType::Int, 0x00495103,
     258, 0x80000000 | 256, 0x00495009,

 // properties: notify_signal_id
       0,
       0,
       0,
       0,
       0,
       8,
       9,
       7,
      76,
      77,
      78,
      79,
      75,
      74,
       5,
       6,
      10,
       4,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      55,
      51,
      58,
      59,
      60,
      61,
      62,
      62,
      62,
      62,
      63,
      64,
      65,
      67,
      68,
      69,
      70,
      71,
      72,
      73,

 // enums: name, flags, count, data
     259, 0x0,   26, 1326,
     249, 0x0,   34, 1378,

 // enum data: key, value
     260, uint(Vehicle::SysStatusSensor3dGyro),
     261, uint(Vehicle::SysStatusSensor3dAccel),
     262, uint(Vehicle::SysStatusSensor3dMag),
     263, uint(Vehicle::SysStatusSensorAbsolutePressure),
     264, uint(Vehicle::SysStatusSensorDifferentialPressure),
     265, uint(Vehicle::SysStatusSensorGPS),
     266, uint(Vehicle::SysStatusSensorOpticalFlow),
     267, uint(Vehicle::SysStatusSensorVisionPosition),
     268, uint(Vehicle::SysStatusSensorLaserPosition),
     269, uint(Vehicle::SysStatusSensorExternalGroundTruth),
     270, uint(Vehicle::SysStatusSensorAngularRateControl),
     271, uint(Vehicle::SysStatusSensorAttitudeStabilization),
     272, uint(Vehicle::SysStatusSensorYawPosition),
     273, uint(Vehicle::SysStatusSensorZAltitudeControl),
     274, uint(Vehicle::SysStatusSensorXYPositionControl),
     275, uint(Vehicle::SysStatusSensorMotorOutputs),
     276, uint(Vehicle::SysStatusSensorRCReceiver),
     277, uint(Vehicle::SysStatusSensor3dGyro2),
     278, uint(Vehicle::SysStatusSensor3dAccel2),
     279, uint(Vehicle::SysStatusSensor3dMag2),
     280, uint(Vehicle::SysStatusSensorGeoFence),
     281, uint(Vehicle::SysStatusSensorAHRS),
     282, uint(Vehicle::SysStatusSensorTerrain),
     283, uint(Vehicle::SysStatusSensorReverseMotor),
     284, uint(Vehicle::SysStatusSensorLogging),
     285, uint(Vehicle::SysStatusSensorBattery),
     286, uint(Vehicle::MAV_TYPE_GENERIC),
     287, uint(Vehicle::MAV_TYPE_FIXED_WING),
     288, uint(Vehicle::MAV_TYPE_QUADROTOR),
     289, uint(Vehicle::MAV_TYPE_COAXIAL),
     290, uint(Vehicle::MAV_TYPE_HELICOPTER),
     291, uint(Vehicle::MAV_TYPE_ANTENNA_TRACKER),
     292, uint(Vehicle::MAV_TYPE_GCS),
     293, uint(Vehicle::MAV_TYPE_AIRSHIP),
     294, uint(Vehicle::MAV_TYPE_FREE_BALLOON),
     295, uint(Vehicle::MAV_TYPE_ROCKET),
     296, uint(Vehicle::MAV_TYPE_GROUND_ROVER),
     297, uint(Vehicle::MAV_TYPE_SURFACE_BOAT),
     298, uint(Vehicle::MAV_TYPE_SUBMARINE),
     299, uint(Vehicle::MAV_TYPE_HEXAROTOR),
     300, uint(Vehicle::MAV_TYPE_OCTOROTOR),
     301, uint(Vehicle::MAV_TYPE_TRICOPTER),
     302, uint(Vehicle::MAV_TYPE_FLAPPING_WING),
     303, uint(Vehicle::MAV_TYPE_KITE),
     304, uint(Vehicle::MAV_TYPE_ONBOARD_CONTROLLER),
     305, uint(Vehicle::MAV_TYPE_VTOL_DUOROTOR),
     306, uint(Vehicle::MAV_TYPE_VTOL_QUADROTOR),
     307, uint(Vehicle::MAV_TYPE_VTOL_TILTROTOR),
     308, uint(Vehicle::MAV_TYPE_VTOL_RESERVED2),
     309, uint(Vehicle::MAV_TYPE_VTOL_RESERVED3),
     310, uint(Vehicle::MAV_TYPE_VTOL_RESERVED4),
     311, uint(Vehicle::MAV_TYPE_VTOL_RESERVED5),
     312, uint(Vehicle::MAV_TYPE_GIMBAL),
     313, uint(Vehicle::MAV_TYPE_ADSB),
     314, uint(Vehicle::MAV_TYPE_PARAFOIL),
     315, uint(Vehicle::MAV_TYPE_DODECAROTOR),
     316, uint(Vehicle::MAV_TYPE_CAMERA),
     317, uint(Vehicle::MAV_TYPE_CHARGING_STATION),
     318, uint(Vehicle::MAV_TYPE_FLARM),
     319, uint(Vehicle::MAV_TYPE_ENUM_END),

       0        // eod
};

void Vehicle::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Vehicle *_t = static_cast<Vehicle *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->missingParametersChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->loadProgressChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 2: _t->mavlinkMessageReceived((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 3: _t->mavCommandResult((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4])),(*reinterpret_cast< bool(*)>(_a[5]))); break;
        case 4: _t->homePositionChanged((*reinterpret_cast< const QGeoCoordinate(*)>(_a[1]))); break;
        case 5: _t->armedChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->landedChanged(); break;
        case 7: _t->flightModeChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 8: _t->flightModesChanged(); break;
        case 9: _t->flightModesOnAirChanged(); break;
        case 10: _t->coordinateChanged((*reinterpret_cast< const QGeoCoordinate(*)>(_a[1]))); break;
        case 11: _t->homePositionReceivedChanged(); break;
        case 12: _t->messagesReceivedChanged(); break;
        case 13: _t->messagesSentChanged(); break;
        case 14: _t->messagesLostChanged(); break;
        case 15: _t->remoteControlRSSIChanged((*reinterpret_cast< uint8_t(*)>(_a[1]))); break;
        case 16: _t->mavlinkRawImu((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 17: _t->mavlinkScaledImu1((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 18: _t->mavlinkScaledImu2((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 19: _t->mavlinkScaledImu3((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 20: _t->rollChanged(); break;
        case 21: _t->pitchChanged(); break;
        case 22: _t->headingChanged(); break;
        case 23: _t->airSpeedChanged(); break;
        case 24: _t->altitudeRelativeChanged(); break;
        case 25: _t->engineSensor_1Changed(); break;
        case 26: _t->engineSensor_2Changed(); break;
        case 27: _t->postionChanged(); break;
        case 28: _t->gpsChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 29: _t->linkChanged(); break;
        case 30: _t->ekfChanged(); break;
        case 31: _t->vibeChanged(); break;
        case 32: _t->headingToHomeChanged(); break;
        case 33: _t->distanceToHomeChanged(); break;
        case 34: _t->currentWaypointChanged(); break;
        case 35: _t->distanceToCurrentWaypointChanged(); break;
        case 36: _t->batteryVoltageChanged(); break;
        case 37: _t->batteryAmpeChanged(); break;
        case 38: _t->groundSpeedChanged(); break;
        case 39: _t->climbSpeedChanged(); break;
        case 40: _t->altitudeAMSLChanged(); break;
        case 41: _t->altitudeAGLChanged(); break;
        case 42: _t->latGPSChanged(); break;
        case 43: _t->lonGPSChanged(); break;
        case 44: _t->hdopGPSChanged(); break;
        case 45: _t->vdopGPSChanged(); break;
        case 46: _t->courseOverGroundGPSChanged(); break;
        case 47: _t->countGPSChanged(); break;
        case 48: _t->lockGPSChanged(); break;
        case 49: _t->fuelUsedChanged(); break;
        case 50: _t->fuelAvailbleChanged(); break;
        case 51: _t->uasChanged(); break;
        case 52: _t->firmwareVersionChanged(); break;
        case 53: _t->firmwareCustomVersionChanged(); break;
        case 54: _t->vehicleUIDChanged(); break;
        case 55: _t->messageSecurityChanged(); break;
        case 56: _t->_sendMessageOnLinkOnThread((*reinterpret_cast< IOFlightController*(*)>(_a[1])),(*reinterpret_cast< mavlink_message_t(*)>(_a[2]))); break;
        case 57: _t->textMessageReceived((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< QString(*)>(_a[4]))); break;
        case 58: _t->unhealthySensorsChanged(); break;
        case 59: _t->sensorsPresentBitsChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 60: _t->sensorsEnabledBitsChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 61: _t->sensorsHealthBitsChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 62: _t->mavlinkStatusChanged(); break;
        case 63: _t->vehicleTypeChanged(); break;
        case 64: _t->paramAirSpeedChanged(); break;
        case 65: _t->paramLoiterRadiusChanged(); break;
        case 66: _t->paramChanged((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 67: _t->rssiChanged(); break;
        case 68: _t->pressABSChanged(); break;
        case 69: _t->sonarRangeChanged(); break;
        case 70: _t->temperatureChanged(); break;
        case 71: _t->propertiesModelChanged(); break;
        case 72: _t->propertiesShowCountChanged(); break;
        case 73: _t->paramsModelChanged(); break;
        case 74: _t->picChanged(); break;
        case 75: _t->useJoystickChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 76: _t->rcinChan1Changed(); break;
        case 77: _t->rcinChan2Changed(); break;
        case 78: _t->rcinChan3Changed(); break;
        case 79: _t->rcinChan4Changed(); break;
        case 80: _t->_loadDefaultParamsShow(); break;
        case 81: _t->_setPropertyValue((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 82: _t->_sendMessageOnLink((*reinterpret_cast< IOFlightController*(*)>(_a[1])),(*reinterpret_cast< mavlink_message_t(*)>(_a[2]))); break;
        case 83: _t->_mavlinkMessageReceived((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 84: _t->_sendGCSHeartbeat(); break;
        case 85: _t->_checkCameraLink(); break;
        case 86: _t->_sendGetParams(); break;
        case 87: _t->_sendQGCTimeToVehicle(); break;
        case 88: _t->requestDataStream((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 89: _t->requestDataStream((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 90: _t->_startPlanRequest(); break;
        case 91: _t->_mavlinkMessageStatus((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< uint64_t(*)>(_a[2])),(*reinterpret_cast< uint64_t(*)>(_a[3])),(*reinterpret_cast< uint64_t(*)>(_a[4])),(*reinterpret_cast< float(*)>(_a[5]))); break;
        case 92: _t->handlePIC(); break;
        case 93: _t->handleUseJoystick((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 94: _t->commandLoiterRadius((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 95: _t->commandRTL(); break;
        case 96: _t->commandLand(); break;
        case 97: _t->commandTakeoff((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 98: { double _r = _t->minimumTakeoffAltitude();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 99: _t->commandGotoLocation((*reinterpret_cast< const QGeoCoordinate(*)>(_a[1]))); break;
        case 100: _t->commandChangeAltitude((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 101: _t->commandSetAltitude((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 102: _t->commandChangeSpeed((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 103: _t->commandOrbit((*reinterpret_cast< const QGeoCoordinate(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3]))); break;
        case 104: _t->pauseVehicle(); break;
        case 105: _t->emergencyStop(); break;
        case 106: _t->abortLanding((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 107: _t->startMission(); break;
        case 108: _t->startEngine(); break;
        case 109: _t->setCurrentMissionSequence((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 110: _t->rebootVehicle(); break;
        case 111: _t->clearMessages(); break;
        case 112: _t->triggerCamera(); break;
        case 113: _t->sendPlan((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 114: { int _r = _t->versionCompare((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 115: { int _r = _t->versionCompare((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 116: _t->motorTest((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 117: _t->setHomeLocation((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 118: _t->setAltitudeRTL((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 119: _t->sendHomePosition((*reinterpret_cast< QGeoCoordinate(*)>(_a[1]))); break;
        case 120: _t->activeProperty((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 121: { int _r = _t->countActiveProperties();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 122: _t->setArmed((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 123: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< double(*)>(_a[7])),(*reinterpret_cast< double(*)>(_a[8])),(*reinterpret_cast< double(*)>(_a[9])),(*reinterpret_cast< double(*)>(_a[10]))); break;
        case 124: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< double(*)>(_a[7])),(*reinterpret_cast< double(*)>(_a[8])),(*reinterpret_cast< double(*)>(_a[9]))); break;
        case 125: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< double(*)>(_a[7])),(*reinterpret_cast< double(*)>(_a[8]))); break;
        case 126: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< double(*)>(_a[7]))); break;
        case 127: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6]))); break;
        case 128: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5]))); break;
        case 129: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4]))); break;
        case 130: _t->sendCommand((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 4:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
            }
            break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
            }
            break;
        case 16:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 17:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 18:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 19:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 56:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< IOFlightController* >(); break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 82:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< IOFlightController* >(); break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 83:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 99:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
            }
            break;
        case 103:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
            }
            break;
        case 119:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Vehicle::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::missingParametersChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::loadProgressChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavlinkMessageReceived)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(int , int , int , int , bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavCommandResult)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(const QGeoCoordinate & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::homePositionChanged)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::armedChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::landedChanged)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::flightModeChanged)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::flightModesChanged)) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::flightModesOnAirChanged)) {
                *result = 9;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(const QGeoCoordinate & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::coordinateChanged)) {
                *result = 10;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::homePositionReceivedChanged)) {
                *result = 11;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::messagesReceivedChanged)) {
                *result = 12;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::messagesSentChanged)) {
                *result = 13;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::messagesLostChanged)) {
                *result = 14;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(uint8_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::remoteControlRSSIChanged)) {
                *result = 15;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavlinkRawImu)) {
                *result = 16;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavlinkScaledImu1)) {
                *result = 17;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavlinkScaledImu2)) {
                *result = 18;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavlinkScaledImu3)) {
                *result = 19;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::rollChanged)) {
                *result = 20;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::pitchChanged)) {
                *result = 21;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::headingChanged)) {
                *result = 22;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::airSpeedChanged)) {
                *result = 23;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::altitudeRelativeChanged)) {
                *result = 24;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::engineSensor_1Changed)) {
                *result = 25;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::engineSensor_2Changed)) {
                *result = 26;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::postionChanged)) {
                *result = 27;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::gpsChanged)) {
                *result = 28;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::linkChanged)) {
                *result = 29;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::ekfChanged)) {
                *result = 30;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::vibeChanged)) {
                *result = 31;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::headingToHomeChanged)) {
                *result = 32;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::distanceToHomeChanged)) {
                *result = 33;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::currentWaypointChanged)) {
                *result = 34;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::distanceToCurrentWaypointChanged)) {
                *result = 35;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::batteryVoltageChanged)) {
                *result = 36;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::batteryAmpeChanged)) {
                *result = 37;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::groundSpeedChanged)) {
                *result = 38;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::climbSpeedChanged)) {
                *result = 39;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::altitudeAMSLChanged)) {
                *result = 40;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::altitudeAGLChanged)) {
                *result = 41;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::latGPSChanged)) {
                *result = 42;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::lonGPSChanged)) {
                *result = 43;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::hdopGPSChanged)) {
                *result = 44;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::vdopGPSChanged)) {
                *result = 45;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::courseOverGroundGPSChanged)) {
                *result = 46;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::countGPSChanged)) {
                *result = 47;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::lockGPSChanged)) {
                *result = 48;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::fuelUsedChanged)) {
                *result = 49;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::fuelAvailbleChanged)) {
                *result = 50;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::uasChanged)) {
                *result = 51;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::firmwareVersionChanged)) {
                *result = 52;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::firmwareCustomVersionChanged)) {
                *result = 53;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::vehicleUIDChanged)) {
                *result = 54;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::messageSecurityChanged)) {
                *result = 55;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(IOFlightController * , mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::_sendMessageOnLinkOnThread)) {
                *result = 56;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(int , int , int , QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::textMessageReceived)) {
                *result = 57;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::unhealthySensorsChanged)) {
                *result = 58;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::sensorsPresentBitsChanged)) {
                *result = 59;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::sensorsEnabledBitsChanged)) {
                *result = 60;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::sensorsHealthBitsChanged)) {
                *result = 61;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::mavlinkStatusChanged)) {
                *result = 62;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::vehicleTypeChanged)) {
                *result = 63;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::paramAirSpeedChanged)) {
                *result = 64;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::paramLoiterRadiusChanged)) {
                *result = 65;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::paramChanged)) {
                *result = 66;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::rssiChanged)) {
                *result = 67;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::pressABSChanged)) {
                *result = 68;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::sonarRangeChanged)) {
                *result = 69;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::temperatureChanged)) {
                *result = 70;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::propertiesModelChanged)) {
                *result = 71;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::propertiesShowCountChanged)) {
                *result = 72;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::paramsModelChanged)) {
                *result = 73;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::picChanged)) {
                *result = 74;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::useJoystickChanged)) {
                *result = 75;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::rcinChan1Changed)) {
                *result = 76;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::rcinChan2Changed)) {
                *result = 77;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::rcinChan3Changed)) {
                *result = 78;
                return;
            }
        }
        {
            using _t = void (Vehicle::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Vehicle::rcinChan4Changed)) {
                *result = 79;
                return;
            }
        }
    } else if (_c == QMetaObject::RegisterPropertyMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< IOFlightController* >(); break;
        case 4:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< ParamsController* >(); break;
        case 3:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< PlanController* >(); break;
        case 17:
        case 16:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
        case 49:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< UAS* >(); break;
        case 0:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< Vehicle* >(); break;
        }
    }

#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        Vehicle *_t = static_cast<Vehicle *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< Vehicle**>(_v) = _t->uav(); break;
        case 1: *reinterpret_cast< JoystickThreaded**>(_v) = _t->joystick(); break;
        case 2: *reinterpret_cast< IOFlightController**>(_v) = _t->communication(); break;
        case 3: *reinterpret_cast< PlanController**>(_v) = _t->planController(); break;
        case 4: *reinterpret_cast< ParamsController**>(_v) = _t->paramsController(); break;
        case 5: *reinterpret_cast< QStringList*>(_v) = _t->flightModes(); break;
        case 6: *reinterpret_cast< QStringList*>(_v) = _t->flightModesOnAir(); break;
        case 7: *reinterpret_cast< QString*>(_v) = _t->flightMode(); break;
        case 8: *reinterpret_cast< float*>(_v) = _t->rcinChan1(); break;
        case 9: *reinterpret_cast< float*>(_v) = _t->rcinChan2(); break;
        case 10: *reinterpret_cast< float*>(_v) = _t->rcinChan3(); break;
        case 11: *reinterpret_cast< float*>(_v) = _t->rcinChan4(); break;
        case 12: *reinterpret_cast< bool*>(_v) = _t->useJoystick(); break;
        case 13: *reinterpret_cast< bool*>(_v) = _t->pic(); break;
        case 14: *reinterpret_cast< bool*>(_v) = _t->armed(); break;
        case 15: *reinterpret_cast< bool*>(_v) = _t->landed(); break;
        case 16: *reinterpret_cast< QGeoCoordinate*>(_v) = _t->coordinate(); break;
        case 17: *reinterpret_cast< QGeoCoordinate*>(_v) = _t->homePosition(); break;
        case 18: *reinterpret_cast< float*>(_v) = _t->roll(); break;
        case 19: *reinterpret_cast< float*>(_v) = _t->pitch(); break;
        case 20: *reinterpret_cast< float*>(_v) = _t->heading(); break;
        case 21: *reinterpret_cast< float*>(_v) = _t->airSpeed(); break;
        case 22: *reinterpret_cast< float*>(_v) = _t->altitudeRelative(); break;
        case 23: *reinterpret_cast< float*>(_v) = _t->engineSensor_1(); break;
        case 24: *reinterpret_cast< float*>(_v) = _t->engineSensor_2(); break;
        case 25: *reinterpret_cast< bool*>(_v) = _t->gpsSignal(); break;
        case 26: *reinterpret_cast< bool*>(_v) = _t->link(); break;
        case 27: *reinterpret_cast< QString*>(_v) = _t->ekfSignal(); break;
        case 28: *reinterpret_cast< QString*>(_v) = _t->vibeSignal(); break;
        case 29: *reinterpret_cast< float*>(_v) = _t->headingToHome(); break;
        case 30: *reinterpret_cast< float*>(_v) = _t->distanceToHome(); break;
        case 31: *reinterpret_cast< int*>(_v) = _t->currentWaypoint(); break;
        case 32: *reinterpret_cast< float*>(_v) = _t->distanceToCurrentWaypoint(); break;
        case 33: *reinterpret_cast< float*>(_v) = _t->batteryVoltage(); break;
        case 34: *reinterpret_cast< float*>(_v) = _t->batteryAmpe(); break;
        case 35: *reinterpret_cast< float*>(_v) = _t->groundSpeed(); break;
        case 36: *reinterpret_cast< float*>(_v) = _t->climbSpeed(); break;
        case 37: *reinterpret_cast< float*>(_v) = _t->altitudeAMSL(); break;
        case 38: *reinterpret_cast< float*>(_v) = _t->altitudeAGL(); break;
        case 39: *reinterpret_cast< float*>(_v) = _t->latGPS(); break;
        case 40: *reinterpret_cast< float*>(_v) = _t->lonGPS(); break;
        case 41: *reinterpret_cast< float*>(_v) = _t->hdopGPS(); break;
        case 42: *reinterpret_cast< float*>(_v) = _t->vdopGPS(); break;
        case 43: *reinterpret_cast< float*>(_v) = _t->courseOverGroundGPS(); break;
        case 44: *reinterpret_cast< int*>(_v) = _t->countGPS(); break;
        case 45: *reinterpret_cast< QString*>(_v) = _t->lockGPS(); break;
        case 46: *reinterpret_cast< float*>(_v) = _t->fuelUsed(); break;
        case 47: *reinterpret_cast< float*>(_v) = _t->fuelAvailble(); break;
        case 48: *reinterpret_cast< QString*>(_v) = _t->messageSecurity(); break;
        case 49: *reinterpret_cast< UAS**>(_v) = _t->uas(); break;
        case 50: *reinterpret_cast< QStringList*>(_v) = _t->unhealthySensors(); break;
        case 51: *reinterpret_cast< int*>(_v) = _t->sensorsPresentBits(); break;
        case 52: *reinterpret_cast< int*>(_v) = _t->sensorsEnabledBits(); break;
        case 53: *reinterpret_cast< int*>(_v) = _t->sensorsHealthBits(); break;
        case 54: *reinterpret_cast< quint64*>(_v) = _t->mavlinkSentCount(); break;
        case 55: *reinterpret_cast< quint64*>(_v) = _t->mavlinkReceivedCount(); break;
        case 56: *reinterpret_cast< quint64*>(_v) = _t->mavlinkLossCount(); break;
        case 57: *reinterpret_cast< float*>(_v) = _t->mavlinkLossPercent(); break;
        case 58: *reinterpret_cast< VEHICLE_MAV_TYPE*>(_v) = _t->vehicleType(); break;
        case 59: *reinterpret_cast< float*>(_v) = _t->paramAirSpeed(); break;
        case 60: *reinterpret_cast< float*>(_v) = _t->paramLoiterRadius(); break;
        case 61: *reinterpret_cast< int*>(_v) = _t->rssi(); break;
        case 62: *reinterpret_cast< float*>(_v) = _t->pressABS(); break;
        case 63: *reinterpret_cast< float*>(_v) = _t->sonarRange(); break;
        case 64: *reinterpret_cast< int*>(_v) = _t->temperature(); break;
        case 65: *reinterpret_cast< QQmlListProperty<Fact>*>(_v) = _t->propertiesModel(); break;
        case 66: *reinterpret_cast< int*>(_v) = _t->propertiesShowCount(); break;
        case 67: *reinterpret_cast< QQmlListProperty<Fact>*>(_v) = _t->paramsModel(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        Vehicle *_t = static_cast<Vehicle *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setUav(*reinterpret_cast< Vehicle**>(_v)); break;
        case 1: _t->setJoystick(*reinterpret_cast< JoystickThreaded**>(_v)); break;
        case 2: _t->setCommunication(*reinterpret_cast< IOFlightController**>(_v)); break;
        case 3: _t->setPlanController(*reinterpret_cast< PlanController**>(_v)); break;
        case 4: _t->setParamsController(*reinterpret_cast< ParamsController**>(_v)); break;
        case 7: _t->setFlightMode(*reinterpret_cast< QString*>(_v)); break;
        case 12: _t->setUseJoystick(*reinterpret_cast< bool*>(_v)); break;
        case 17: _t->setHomePosition(*reinterpret_cast< QGeoCoordinate*>(_v)); break;
        case 26: _t->setLink(*reinterpret_cast< bool*>(_v)); break;
        case 27: _t->setEkfSignal(*reinterpret_cast< QString*>(_v)); break;
        case 28: _t->setVibeSignal(*reinterpret_cast< QString*>(_v)); break;
        case 48: _t->setMessageSecurity(*reinterpret_cast< QString*>(_v)); break;
        case 58: _t->setVehicleType(*reinterpret_cast< VEHICLE_MAV_TYPE*>(_v)); break;
        case 59: _t->setParamAirSpeed(*reinterpret_cast< float*>(_v)); break;
        case 60: _t->setParamLoiterRadius(*reinterpret_cast< float*>(_v)); break;
        case 66: _t->setPropertiesShowCount(*reinterpret_cast< int*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject Vehicle::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_Vehicle.data,
      qt_meta_data_Vehicle,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *Vehicle::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Vehicle::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Vehicle.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Vehicle::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 131)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 131;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 131)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 131;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 68;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 68;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 68;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 68;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 68;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 68;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void Vehicle::missingParametersChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Vehicle::loadProgressChanged(float _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Vehicle::mavlinkMessageReceived(mavlink_message_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Vehicle::mavCommandResult(int _t1, int _t2, int _t3, int _t4, bool _t5)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)), const_cast<void*>(reinterpret_cast<const void*>(&_t5)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void Vehicle::homePositionChanged(const QGeoCoordinate & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void Vehicle::armedChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void Vehicle::landedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}

// SIGNAL 7
void Vehicle::flightModeChanged(const QString & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void Vehicle::flightModesChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 8, nullptr);
}

// SIGNAL 9
void Vehicle::flightModesOnAirChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 9, nullptr);
}

// SIGNAL 10
void Vehicle::coordinateChanged(const QGeoCoordinate & _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 10, _a);
}

// SIGNAL 11
void Vehicle::homePositionReceivedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 11, nullptr);
}

// SIGNAL 12
void Vehicle::messagesReceivedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 12, nullptr);
}

// SIGNAL 13
void Vehicle::messagesSentChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 13, nullptr);
}

// SIGNAL 14
void Vehicle::messagesLostChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 14, nullptr);
}

// SIGNAL 15
void Vehicle::remoteControlRSSIChanged(uint8_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 15, _a);
}

// SIGNAL 16
void Vehicle::mavlinkRawImu(mavlink_message_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 16, _a);
}

// SIGNAL 17
void Vehicle::mavlinkScaledImu1(mavlink_message_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 17, _a);
}

// SIGNAL 18
void Vehicle::mavlinkScaledImu2(mavlink_message_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 18, _a);
}

// SIGNAL 19
void Vehicle::mavlinkScaledImu3(mavlink_message_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 19, _a);
}

// SIGNAL 20
void Vehicle::rollChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 20, nullptr);
}

// SIGNAL 21
void Vehicle::pitchChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 21, nullptr);
}

// SIGNAL 22
void Vehicle::headingChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 22, nullptr);
}

// SIGNAL 23
void Vehicle::airSpeedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 23, nullptr);
}

// SIGNAL 24
void Vehicle::altitudeRelativeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 24, nullptr);
}

// SIGNAL 25
void Vehicle::engineSensor_1Changed()
{
    QMetaObject::activate(this, &staticMetaObject, 25, nullptr);
}

// SIGNAL 26
void Vehicle::engineSensor_2Changed()
{
    QMetaObject::activate(this, &staticMetaObject, 26, nullptr);
}

// SIGNAL 27
void Vehicle::postionChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 27, nullptr);
}

// SIGNAL 28
void Vehicle::gpsChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 28, _a);
}

// SIGNAL 29
void Vehicle::linkChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 29, nullptr);
}

// SIGNAL 30
void Vehicle::ekfChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 30, nullptr);
}

// SIGNAL 31
void Vehicle::vibeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 31, nullptr);
}

// SIGNAL 32
void Vehicle::headingToHomeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 32, nullptr);
}

// SIGNAL 33
void Vehicle::distanceToHomeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 33, nullptr);
}

// SIGNAL 34
void Vehicle::currentWaypointChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 34, nullptr);
}

// SIGNAL 35
void Vehicle::distanceToCurrentWaypointChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 35, nullptr);
}

// SIGNAL 36
void Vehicle::batteryVoltageChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 36, nullptr);
}

// SIGNAL 37
void Vehicle::batteryAmpeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 37, nullptr);
}

// SIGNAL 38
void Vehicle::groundSpeedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 38, nullptr);
}

// SIGNAL 39
void Vehicle::climbSpeedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 39, nullptr);
}

// SIGNAL 40
void Vehicle::altitudeAMSLChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 40, nullptr);
}

// SIGNAL 41
void Vehicle::altitudeAGLChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 41, nullptr);
}

// SIGNAL 42
void Vehicle::latGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 42, nullptr);
}

// SIGNAL 43
void Vehicle::lonGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 43, nullptr);
}

// SIGNAL 44
void Vehicle::hdopGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 44, nullptr);
}

// SIGNAL 45
void Vehicle::vdopGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 45, nullptr);
}

// SIGNAL 46
void Vehicle::courseOverGroundGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 46, nullptr);
}

// SIGNAL 47
void Vehicle::countGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 47, nullptr);
}

// SIGNAL 48
void Vehicle::lockGPSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 48, nullptr);
}

// SIGNAL 49
void Vehicle::fuelUsedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 49, nullptr);
}

// SIGNAL 50
void Vehicle::fuelAvailbleChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 50, nullptr);
}

// SIGNAL 51
void Vehicle::uasChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 51, nullptr);
}

// SIGNAL 52
void Vehicle::firmwareVersionChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 52, nullptr);
}

// SIGNAL 53
void Vehicle::firmwareCustomVersionChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 53, nullptr);
}

// SIGNAL 54
void Vehicle::vehicleUIDChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 54, nullptr);
}

// SIGNAL 55
void Vehicle::messageSecurityChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 55, nullptr);
}

// SIGNAL 56
void Vehicle::_sendMessageOnLinkOnThread(IOFlightController * _t1, mavlink_message_t _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 56, _a);
}

// SIGNAL 57
void Vehicle::textMessageReceived(int _t1, int _t2, int _t3, QString _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 57, _a);
}

// SIGNAL 58
void Vehicle::unhealthySensorsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 58, nullptr);
}

// SIGNAL 59
void Vehicle::sensorsPresentBitsChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 59, _a);
}

// SIGNAL 60
void Vehicle::sensorsEnabledBitsChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 60, _a);
}

// SIGNAL 61
void Vehicle::sensorsHealthBitsChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 61, _a);
}

// SIGNAL 62
void Vehicle::mavlinkStatusChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 62, nullptr);
}

// SIGNAL 63
void Vehicle::vehicleTypeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 63, nullptr);
}

// SIGNAL 64
void Vehicle::paramAirSpeedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 64, nullptr);
}

// SIGNAL 65
void Vehicle::paramLoiterRadiusChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 65, nullptr);
}

// SIGNAL 66
void Vehicle::paramChanged(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 66, _a);
}

// SIGNAL 67
void Vehicle::rssiChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 67, nullptr);
}

// SIGNAL 68
void Vehicle::pressABSChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 68, nullptr);
}

// SIGNAL 69
void Vehicle::sonarRangeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 69, nullptr);
}

// SIGNAL 70
void Vehicle::temperatureChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 70, nullptr);
}

// SIGNAL 71
void Vehicle::propertiesModelChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 71, nullptr);
}

// SIGNAL 72
void Vehicle::propertiesShowCountChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 72, nullptr);
}

// SIGNAL 73
void Vehicle::paramsModelChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 73, nullptr);
}

// SIGNAL 74
void Vehicle::picChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 74, nullptr);
}

// SIGNAL 75
void Vehicle::useJoystickChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 75, _a);
}

// SIGNAL 76
void Vehicle::rcinChan1Changed()
{
    QMetaObject::activate(this, &staticMetaObject, 76, nullptr);
}

// SIGNAL 77
void Vehicle::rcinChan2Changed()
{
    QMetaObject::activate(this, &staticMetaObject, 77, nullptr);
}

// SIGNAL 78
void Vehicle::rcinChan3Changed()
{
    QMetaObject::activate(this, &staticMetaObject, 78, nullptr);
}

// SIGNAL 79
void Vehicle::rcinChan4Changed()
{
    QMetaObject::activate(this, &staticMetaObject, 79, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
