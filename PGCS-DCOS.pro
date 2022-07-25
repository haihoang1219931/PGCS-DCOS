QT += qml quick multimedia network positioning sensors core gui serialport charts widgets
QT += webengine
QT += dbus
CONFIG += c++11 no_keywords console

RESOURCES += qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model

#message($${QML_IMPORT_PATH})
# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
include(thirdparty/qsyncable/qsyncable.pri)
HOME = $$system(echo $HOME)
message($$HOME)
CONFIG += use_utils

CONFIG += use_flight_control

#CONFIG += use_ucapi

CONFIG += use_camera_control

#CONFIG += use_video_gpu

CONFIG += use_video_cpu

#CONFIG += use_line_detector

QML_IMPORT_PATH += \
    $$PWD \
    $$PWD/src/QmlControls \
    $$PWD/src/Cpp/Flight
SOURCES += \
    main.cpp
INCLUDEPATH += $$PWD/src/Cpp

use_utils{
SOURCES += \
    src/Cpp/Utils/Bytes/ByteManipulation.cpp \
    src/Cpp/Utils/Files/FileControler.cpp \
    src/Cpp/Utils/Files/PlateLog.cpp \
    src/Cpp/Utils/Files/PlateLogThread.cpp \
    src/Cpp/Utils/Joystick/JoystickController.cpp \
    src/Cpp/Utils/Joystick/JoystickTask.cpp \
    src/Cpp/Utils/Joystick/JoystickThreaded.cpp \
    src/Cpp/Utils/Setting/tinyxml2.cpp \
    src/Cpp/Utils/Setting/config.cpp \
    src/Cpp/Utils/Maplib/Marker/Marker.cpp \
    src/Cpp/Utils/Maplib/Marker/MarkerList.cpp \
    src/Cpp/Utils/Maplib/Elevation.cpp \
    src/Cpp/Utils/Machine/computer.cpp \
    src/Cpp/Utils/Machine/ConnectionChecking.cpp \
    src/Cpp/Utils/Machine/ConnectionThread.cpp \
    src/Cpp/Utils/Maplib/profilepath.cpp \
    src/Cpp/Utils/Log/LogController.cpp \
    src/Cpp/Utils/Maplib/Model/symbol.cpp \
    src/Cpp/Utils/Maplib/Model/symbolmodel.cpp\
    src/Cpp/Utils/Network/NetworkInfo.cpp \
    src/Cpp/Utils/Network/NetworkManager.cpp
HEADERS += \
    src/Cpp/Utils/Bytes/ByteManipulation.h \
    src/Cpp/Utils/Files/FileControler.h \
    src/Cpp/Utils/Files/PlateLog.h \
    src/Cpp/Utils/Files/PlateLogThread.h \
    src/Cpp/Utils/Joystick/JoystickController.h \
    src/Cpp/Utils/Joystick/JoystickTask.h \
    src/Cpp/Utils/Joystick/JoystickThreaded.h \
    src/Cpp/Utils/Setting/tinyxml2.h \
    src/Cpp/Utils/Setting/config.h \
    src/Cpp/Utils/Maplib/Marker/Marker.h \
    src/Cpp/Utils/Maplib/Marker/MarkerList.h \
    src/Cpp/Utils/Maplib/Elevation.h \
    src/Cpp/Utils/Machine/computer.hpp \
    src/Cpp/Utils/Machine/ConnectionChecking.h \
    src/Cpp/Utils/Machine/ConnectionThread.h \
    src/Cpp/Utils/Maplib/profilepath.h \
    src/Cpp/Utils/Log/LogController.h \
    src/Cpp/Utils/Maplib/Model/symbol.h \
    src/Cpp/Utils/Maplib/Model/symbolmodel.h \
    src/Cpp/Utils/Network/NetworkInfo.h \
    src/Cpp/Utils/Network/NetworkManager.h
}
# Flight controller
use_flight_control{
INCLUDEPATH += $$PWD/src/Cpp/Flight/Protocols/mavlink/v2.0
INCLUDEPATH += $$PWD/src/Cpp/Flight/Protocols/mavlink/v2.0/ardupilotmega

SOURCES += \
    src/Cpp/Flight/Com/IOFlightController.cpp \
    src/Cpp/Flight/Com/LinkInterface.cpp \
    src/Cpp/Flight/Com/LinkInterfaceManager.cpp \
    src/Cpp/Flight/Com/MessageManager.cpp \
    src/Cpp/Flight/Com/QGCMAVLink.cc \
    src/Cpp/Flight/Com/TCPLink.cpp \
    src/Cpp/Flight/Com/SerialLink.cpp \
    src/Cpp/Flight/Com/UDPMulticastLink.cpp \
    src/Cpp/Flight/Com/UDPLink.cpp \
    src/Cpp/Flight/Com/ProtocolInterface.cpp \
    src/Cpp/Flight/Com/MavlinkProtocol.cpp \
    src/Cpp/Flight/Firmware/APM/ArduCopterFirmware.cpp \
    src/Cpp/Flight/Firmware/APM/QuadPlaneFirmware.cpp \
    src/Cpp/Flight/Firmware/FirmwarePlugin.cpp \
    src/Cpp/Flight/Firmware/FirmwarePluginManager.cpp \
    src/Cpp/Flight/Mission/MissionController.cpp \
    src/Cpp/Flight/Mission/MissionItem.cpp \
    src/Cpp/Flight/Mission/PlanController.cpp \
    src/Cpp/Flight/Vehicle/Vehicle.cpp \
    src/Cpp/Flight/Params/ParamsController.cpp \
    src/Cpp/Flight/UAS/UAS.cpp \
    src/Cpp/Flight/UAS/UASMessage.cpp \
    src/Cpp/Flight/Params/Fact.cpp \
    src/Cpp/Flight/Telemetry/TelemetryController.cpp

HEADERS += \
    src/Cpp/Flight/Com/IOFlightController.h \
    src/Cpp/Flight/Com/LinkInterface.h \
    src/Cpp/Flight/Com/LinkInterfaceManager.h \
    src/Cpp/Flight/Com/MessageManager.h \
    src/Cpp/Flight/Com/QGCMAVLink.h \
    src/Cpp/Flight/Com/TCPLink.h \
    src/Cpp/Flight/Com/SerialLink.h \
    src/Cpp/Flight/Com/UDPMulticastLink.h \
    src/Cpp/Flight/Com/UDPLink.h \
    src/Cpp/Flight/Com/ProtocolInterface.h \
    src/Cpp/Flight/Com/MavlinkProtocol.h \
    src/Cpp/Flight/Firmware/APM/ArduCopterFirmware.h \
    src/Cpp/Flight/Firmware/APM/QuadPlaneFirmware.h \
    src/Cpp/Flight/Firmware/FirmwarePlugin.h \
    src/Cpp/Flight/Firmware/FirmwarePluginManager.h \
    src/Cpp/Flight/Mission/MissionController.h \
    src/Cpp/Flight/Mission/MissionItem.h \
    src/Cpp/Flight/Mission/PlanController.h \
    src/Cpp/Flight/Vehicle/Vehicle.h \
    src/Cpp/Flight/Params/ParamsController.h \
    src/Cpp/Flight/UAS/UAS.h \
    src/Cpp/Flight/UAS/UASMessage.h \
    src/Cpp/Flight/Params/Fact.h \
    src/Cpp/Flight/Telemetry/TelemetryController.h
}
# UC libs KURENTO
use_ucapi{
DEFINES += UC_API
DEFINES += SIO_TLS
QML_IMPORT_PATH += \
    $$PWD/src/QmlControls/UC
INCLUDEPATH += src/Cpp/UC
SOURCES += \
    src/Cpp/UC/UCDataModel.cpp \
    src/Cpp/UC/UCEventListener.cpp
HEADERS += \
    src/Cpp/UC/UCDataModel.hpp \
    src/Cpp/UC/UCEventListener.hpp

INCLUDEPATH += $$PWD/src/Cpp/UC/sioclient/lib
INCLUDEPATH += $$PWD/src/Cpp/UC/boost1.62/include
INCLUDEPATH += $$PWD/src/Cpp/UC/openssl

LIBS+= -L$$HOME/workspaces/boost1.62/lib -lboost_system -lboost_chrono -lboost_thread -lboost_timer
LIBS += -L/usr/local/lib -lcrypto -lssl
LIBS += -lpthread

SOURCES += \
    src/Cpp/UC/api/app_socket_api.cpp \
    src/Cpp/UC/json/jsoncpp.cpp \
    src/Cpp/UC/sioclient/src/internal/sio_client_impl.cpp \
    src/Cpp/UC/sioclient/src/internal/sio_packet.cpp \
    src/Cpp/UC/sioclient/src/sio_client.cpp \
    src/Cpp/UC/sioclient/src/sio_socket.cpp

HEADERS += \
    src/Cpp/UC/api/app_socket_api.hpp \
    src/Cpp/UC/json/json-forwards.h \
    src/Cpp/UC/json/json.h \
    src/Cpp/UC/sioclient/src/internal/sio_client_impl.h \
    src/Cpp/UC/sioclient/src/internal/sio_packet.h \
    src/Cpp/UC/sioclient/src/sio_client.h \
    src/Cpp/UC/sioclient/src/sio_message.h \
    src/Cpp/UC/sioclient/src/sio_socket.h

}
# Control camera
use_camera_control{
DEFINES += CAMERA_CONTROL
# Opencv
unix:!macx: LIBS += -L/usr/local/lib/  \
    -lopencv_objdetect \
    -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_features2d -lopencv_calib3d \
    -lopencv_videostab \
    -lopencv_video \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lopencv_imgproc \
    -lopencv_videoio
## FFMPEG
INCLUDEPATH += /usr/include/x86_64-linux-gnu/
DEPENDPATH += /usr/include/x86_64-linux-gnu/
LIBS +=  \
    -lavformat \
    -lavcodec \
    -lavutil \
    -lswscale \
    -lswresample
# GStreamer
unix:!macx: DEPENDPATH += /usr/local/include
unix:!macx: INCLUDEPATH += /usr/include/gstreamer-1.0
#unix:!macx: INCLUDEPATH += /usr/local/include/gstreamer-1.0
unix:!macx: INCLUDEPATH += /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include
unix:!macx: INCLUDEPATH += /usr/include/glib-2.0
unix:!macx: INCLUDEPATH += /usr/lib/x86_64-linux-gnu/glib-2.0/include

unix:!macx: LIBS += -LD:\usr\lib\x86_64-linux-gnu\
    -lglib-2.0 \
    -lgstreamer-1.0 \
    -lgstapp-1.0 \
    -lgstrtsp-1.0 \
    -lgstrtspserver-1.0 \
    -lgobject-2.0 \
    -lgstvideo-1.0
unix:!macx: INCLUDEPATH += /usr/local/include
unix:!macx: DEPENDPATH += /usr/local/include
SOURCES += \
    src/Cpp/Payload/CameraController.cpp \
    src/Cpp/Payload/Buffer/BufferOut.cpp \
    src/Cpp/Payload/Cache/Cache.cpp \
    src/Cpp/Payload/VideoEngine/VideoEngineInterface.cpp \
    src/Cpp/Payload/VideoEngine/VRTSPServer.cpp \
    src/Cpp/Payload/VideoEngine/VSavingWorker.cpp \
    src/Cpp/Payload/VideoDisplay/ImageItem.cpp \
    src/Cpp/Payload/VideoDisplay/I420Render.cpp \
    src/Cpp/Payload/VideoDisplay/VideoRender.cpp \
    src/Cpp/Payload/TargetLocation/TargetLocalization.cpp

HEADERS += \
    src/Cpp/Payload/CameraController.h \
    src/Cpp/Payload/Packet/Object.h \
    src/Cpp/Payload/Packet/XPoint.h \
    src/Cpp/Payload/Packet/utils.h \
    src/Cpp/Payload/Packet/Common_type.h \
    src/Cpp/Payload/Buffer/BufferOut.h \
    src/Cpp/Payload/Buffer/RollBuffer.h \
    src/Cpp/Payload/Cache/Cache.h \
    src/Cpp/Payload/Cache/TrackObject.h \
    src/Cpp/Payload/Cache/CacheItem.h \
    src/Cpp/Payload/Cache/DetectedObjectsCacheItem.h \
    src/Cpp/Payload/Cache/FixedMemory.h \
    src/Cpp/Payload/Cache/GstFrameCacheItem.h \
    src/Cpp/Payload/Cache/ProcessImageCacheItem.h \
    src/Cpp/Payload/VideoEngine/VideoEngineInterface.h \
    src/Cpp/Payload/VideoEngine/VRTSPServer.h \
    src/Cpp/Payload/VideoEngine/VSavingWorker.h \
    src/Cpp/Payload/VideoDisplay/ImageItem.h \
    src/Cpp/Payload/VideoDisplay/I420Render.h \
    src/Cpp/Payload/VideoDisplay/VideoRender.h \
    src/Cpp/Payload/TargetLocation/TargetLocalization.h

# Gimbal control
SOURCES += \
    src/Cpp/Payload/GimbalController/CM160/CM160Gimbal.cpp \
    src/Cpp/Payload/GimbalController/CM160/GimbalDiscoverer.cpp \
    src/Cpp/Payload/GimbalController/CM160/GimbalPacketParser.cpp \
    src/Cpp/Payload/GimbalController/CM160/UDPPayload.cpp \
    src/Cpp/Payload/GimbalController/CM160/UDPSenderListener.cpp \
    src/Cpp/Payload/GimbalController/CM160/UavvPacket.cpp \
    src/Cpp/Payload/GimbalController/CM160/UavvPacketHelper.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavCombinedZoomEnable.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavCurrentExposureMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavDefog.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavDisableInfraredCutFilter.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableAutoExposure.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableAutoFocus.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableDigitalZoom.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableEOSensor.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableLensStabilization.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableManualIris.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableManualShutterMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavGetZoomPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavInvertPicture.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSensorCurrentFoV.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSensorZoom.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetCameraGain.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetDigitalZoomPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetDigitalZoomVelocity.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetEOOpticalZoomPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetEOOpticalZoomVelocity.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetEOSensorVideoMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetFocus.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetIris.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetShutterSpeed.cpp \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavZoomPositionResponse.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvAltitudeOffset.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvCurrentCornerLocations.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvCurrentGeolockSetpoit.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvCurrentTargetLocation.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvExternalAltitude.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvExternalPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvGNSSStatus.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvGimbalMisalignmentOffset.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvGimbalOrientationOffset.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvPlatformOrientation.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvPlatformPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvSeedTerrainHeight.cpp \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvSetGeolockLocation.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvCurrentGimbalMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvCurrentGimbalPositionRate.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvInitialiseGimbal.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvPanPositionReply.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSceneSteering.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSceneSteeringConfiguration.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanTiltPosition.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanTiltVelocity.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanVelocity.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPrimaryVideo.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetTiltPositon.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetTiltVelocity.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvStowConfiguration.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvStowMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvStowStatusResponse.cpp \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvTiltPositionReply.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavEnableIRIsotherm.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavIRSensorTemperatureResponse.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavMWIRTempPreset.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavPerformFFC.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavResetIRCamera.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetDynamicDDE.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetFFCMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetFFCTemperatureDelta.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRAGCMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRBrightness.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRBrightnessBias.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRConstrast.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRGainMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRITTMidpoint.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRIsothermThresholds.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRMaxGain.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRPalette.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRPlateauLevel.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRVideoModulation.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRVideoOrientation.cpp \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRZoom.cpp \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/ArmLaserDevice.cpp \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/FireLaserDevice.cpp \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserDeviceStatus.cpp \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserRange.cpp \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserRangeStart.cpp \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserRangeStatus.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvConfigurePacketRates.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvEnableGyroStabilisation.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvEnableMessageACK.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvEnableStreamMode.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvMessageACKResponse.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvNetworkConfiguration.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvRequestResponse.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvSaveParameters.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvSetSystemTime.cpp \
    src/Cpp/Payload/GimbalController/CM160/system/UavvVersion.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocoleosensorpackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolgeopointingpackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolgimbalpackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolirsensorpackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocollaserrangefinderpackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolsystempackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolvideoprocessorpackets.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvChangeVideoRecordingState.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvCurrentImageSize.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvCurrentRecordingState.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvH264StreamParameters.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvModifyObjectTrack.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvModifyTrackIndex.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvNudgeTrack.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvOverlay.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvStabilisationParameters.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvStabiliseOnTrack.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvTakeSnapshot.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvTrackingParameters.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvVideoConfiguration.cpp \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvVideoDestination.cpp \
    src/Cpp/Payload/GimbalController/GimbalController.cpp \
    src/Cpp/Payload/GimbalController/GimbalData.cpp \
    src/Cpp/Payload/GimbalController/GimbalInterface.cpp \
    src/Cpp/Payload/GimbalController/GimbalInterfaceManager.cpp \
    src/Cpp/Payload/GimbalController/Gremsey/GimbalControl.cpp \
    src/Cpp/Payload/GimbalController/Gremsey/GremseyGimbal.cpp \
    src/Cpp/Payload/GimbalController/Gremsey/SensorController.cpp \
    src/Cpp/Payload/GimbalController/Gremsey/SBusGimbal.cpp \
    src/Cpp/Payload/GimbalController/Treron/TreronGimbal.cpp \
    src/Cpp/Payload/GimbalController/Treron/TreronGimbalPacketParser.cpp \
    src/Cpp/Payload/GimbalController/Treron/Command/GeoCommands.cpp \
    src/Cpp/Payload/GimbalController/Treron/Command/IPCCommands.cpp \
    src/Cpp/Payload/GimbalController/Treron/Command/MotionCCommands.cpp \
    src/Cpp/Payload/GimbalController/Treron/Command/SystemCommands.cpp
HEADERS += \
    src/Cpp/Payload/GimbalController/CM160/CM160Gimbal.h \
    src/Cpp/Payload/GimbalController/CM160/GimbalDiscoverer.h \
    src/Cpp/Payload/GimbalController/CM160/GimbalPacketParser.h \
    src/Cpp/Payload/GimbalController/CM160/IPEndpoint.h \
    src/Cpp/Payload/GimbalController/CM160/UDPPayload.h \
    src/Cpp/Payload/GimbalController/CM160/UDPSenderListener.h \
    src/Cpp/Payload/GimbalController/CM160/UavvGimbalProtocol.h \
    src/Cpp/Payload/GimbalController/CM160/UavvPacket.h \
    src/Cpp/Payload/GimbalController/CM160/UavvPacketHelper.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavCombinedZoomEnable.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavCurrentExposureMode.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavDefog.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavDisableInfraredCutFilter.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableAutoExposure.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableAutoFocus.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableDigitalZoom.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableEOSensor.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableLensStabilization.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableManualIris.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavEnableManualShutterMode.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavGetZoomPosition.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavInvertPicture.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSensorCurrentFoV.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSensorZoom.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetCameraGain.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetDigitalZoomPosition.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetDigitalZoomVelocity.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetEOOpticalZoomPosition.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetEOOpticalZoomVelocity.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetEOSensorVideoMode.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetFocus.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetIris.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavSetShutterSpeed.h \
    src/Cpp/Payload/GimbalController/CM160/eosensor/UavZoomPositionResponse.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvAltitudeOffset.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvCurrentCornerLocations.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvCurrentGeolockSetpoit.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvCurrentTargetLocation.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvExternalAltitude.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvExternalPosition.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvGNSSStatus.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvGimbalMisalignmentOffset.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvGimbalOrientationOffset.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvPlatformOrientation.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvPlatformPosition.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvSeedTerrainHeight.h \
    src/Cpp/Payload/GimbalController/CM160/geopointing/UavvSetGeolockLocation.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvCurrentGimbalMode.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvCurrentGimbalPositionRate.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvInitialiseGimbal.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvPanPositionReply.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSceneSteering.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSceneSteeringConfiguration.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanPosition.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanTiltPosition.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanTiltVelocity.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPanVelocity.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetPrimaryVideo.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetTiltPositon.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvSetTiltVelocity.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvStowConfiguration.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvStowMode.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvStowStatusResponse.h \
    src/Cpp/Payload/GimbalController/CM160/gimbal/UavvTiltPositionReply.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavEnableIRIsotherm.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavIRSensorTemperatureResponse.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavMWIRTempPreset.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavPerformFFC.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavResetIRCamera.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetDynamicDDE.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetFFCMode.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetFFCTemperatureDelta.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRAGCMode.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRBrightness.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRBrightnessBias.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRContrast.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRGainMode.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRITTMidpoint.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRIsothermThresholds.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRMaxGain.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRPalette.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRPlateauLevel.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRVideoModulation.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRVideoOrientation.h \
    src/Cpp/Payload/GimbalController/CM160/irsensor/UavSetIRZoom.h \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/ArmLaserDevice.h \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/FireLaserDevice.h \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserDeviceStatus.h \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserRange.h \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserRangeStart.h \
    src/Cpp/Payload/GimbalController/CM160/laserrangefinder/LaserRangeStatus.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvConfigurePacketRates.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvEnableGyroStabilisation.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvEnableMessageACK.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvEnableStreamMode.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvMessageACKResponse.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvNetworkConfiguration.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvRequestResponse.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvSaveParameters.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvSetSystemTime.h \
    src/Cpp/Payload/GimbalController/CM160/system/UavvVersion.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocoleosensorpackets.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolgeopointingpackets.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolgimbalpackets.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolirsensorpackets.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocollaserrangefinderpackets.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolsystempackets.h \
    src/Cpp/Payload/GimbalController/CM160/uavvgimbalprotocolvideoprocessorpackets.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvChangeVideoRecordingState.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvCurrentImageSize.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvCurrentRecordingState.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvH264StreamParameters.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvModifyObjectTrack.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvModifyTrackIndex.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvNudgeTrack.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvOverlay.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvStabilisationParameters.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvStabiliseOnTrack.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvTakeSnapshot.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvTrackingParameters.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvVideoConfiguration.h \
    src/Cpp/Payload/GimbalController/CM160/videoprocessor/UavvVideoDestination.h \
    src/Cpp/Payload/GimbalController/GimbalController.h \
    src/Cpp/Payload/GimbalController/GimbalData.h \
    src/Cpp/Payload/GimbalController/GimbalInterface.h \
    src/Cpp/Payload/GimbalController/GimbalInterfaceManager.h \
    src/Cpp/Payload/GimbalController/Gremsey/GimbalControl.h \
    src/Cpp/Payload/GimbalController/Gremsey/GremseyGimbal.h \
    src/Cpp/Payload/GimbalController/Gremsey/SensorController.h \
    src/Cpp/Payload/GimbalController/Gremsey/SBusGimbal.h \
    src/Cpp/Payload/GimbalController/Treron/TreronGimbal.h \
    src/Cpp/Payload/GimbalController/Treron/TreronGimbalPacketParser.h \
    src/Cpp/Payload/GimbalController/Treron/Command/GeoCommands.h \
    src/Cpp/Payload/GimbalController/Treron/Command/IPCCommands.h \
    src/Cpp/Payload/GimbalController/Treron/Command/MotionCCommands.h \
    src/Cpp/Payload/GimbalController/Treron/Command/SystemCommands.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Common_type.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Confirm.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EnGeoLocation.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EOS.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EyeCheck.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EyeEvent.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EyephoenixProtocol.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EyeRotationMatrix.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/EyeStatus.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/GimbalMode.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/GimbalRecord.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/GimbalRecordStatus.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/GimbalStab.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/GPSData.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/GPSRate.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/HConfigMessage.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/ImageStab.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/InstallMode.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/IPCStatusResponse.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/KLV.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/LockMode.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Matrix.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/MCParams.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/MData.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/MotionAngle.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/MotionCStatus.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/MotionImage.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Object.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/PTAngle.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/PTAngleDiff.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/PTRateFactor.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/RapidView.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/RequestResponsePacket.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/RFData.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/RFRequest.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/RTData.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/SceneSteering.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/ScreenPoint.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/SensorColor.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/SensorId.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Snapshot.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/StreamingProfile.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/SystemStatus.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/TargetPosition.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Telemetry.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/TrackObject.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/TrackResponse.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/TrackSize.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/utils.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/Vector.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/XPoint.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/ZoomData.h \
    src/Cpp/Payload/GimbalController/Treron/Packet/ZoomStatus.h
HEADERS += \
    src/Cpp/Payload/Algorithms/stabilizer/dando_02/stab_gcs_kiir.hpp \
    src/Cpp/Payload/Algorithms/tracker/dando/HTrack/ffttools.hpp \
    src/Cpp/Payload/Algorithms/tracker/dando/HTrack/fhog.hpp \
    src/Cpp/Payload/Algorithms/tracker/dando/HTrack/saliency.h \
    src/Cpp/Payload/Algorithms/tracker/dando/LME/lme.hpp \
    src/Cpp/Payload/Algorithms/tracker/dando/ITrack.hpp \
    src/Cpp/Payload/Algorithms/tracker/dando/Utilities.hpp \
    src/Cpp/Payload/Algorithms/tracker/mosse/tracker.h
SOURCES += \
    src/Cpp/Payload/Algorithms/stabilizer/dando_02/stab_gcs_kiir.cpp \
    src/Cpp/Payload/Algorithms/tracker/dando/HTrack/ffttools.cpp \
    src/Cpp/Payload/Algorithms/tracker/dando/HTrack/fhog.cpp \
    src/Cpp/Payload/Algorithms/tracker/dando/HTrack/saliency.cpp \
    src/Cpp/Payload/Algorithms/tracker/dando/LME/lme.cpp \
    src/Cpp/Payload/Algorithms/tracker/dando/ITrack.cpp \
    src/Cpp/Payload/Algorithms/tracker/dando/Utilities.cpp \
    src/Cpp/Payload/Algorithms/tracker/mosse/tracker.cpp

}


# Image processing based GPU
use_video_gpu{
DEFINES += USE_VIDEO_GPU
# project build directories
DESTDIR     = $$system(pwd)/build
OBJECTS_DIR = $$DESTDIR
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3
# Cuda sources
CUDA_SOURCES += src/Cpp/Payload/GPUBased/Cuda/ipcuda_image.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-10.1
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
# libs used in your code
LIBS += -lcudart -lcuda -lpthread
# GPU architecture
CUDA_ARCH     = sm_61                # Yeah! I've a new device. Adjust with your compute capability
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v


# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME} | sed \"s/^.*: //\"

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

INCLUDEPATH += /usr/local/cuda-10.1/include
INCLUDEPATH += /usr/local/cuda-10.1/targets/x86_64-linux/include

# TensorFlow r1.14
#include(tensorflow_dependency.pri)

INCLUDEPATH += $$HOME/install/tensorflow/tensorflow-1.14.0
INCLUDEPATH += $$HOME/install/tensorflow/tensorflow-1.14.0/tensorflow
INCLUDEPATH += $$HOME/install/tensorflow/tensorflow-1.14.0/bazel-tensorflow-1.14.0/external/eigen_archive
INCLUDEPATH += $$HOME/install/tensorflow/tensorflow-1.14.0/bazel-tensorflow-1.14.0/external/protobuf_archive/src
INCLUDEPATH += $$HOME/install/tensorflow/tensorflow-1.14.0/bazel-genfiles

LIBS += -L$$HOME/install/tensorflow/tensorflow-1.14.0/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework

LIBS += `pkg-config --libs opencv`
# End TensorFlow
LIBS += -LD:\usr\local\lib \
 -ldarknet
message($$LIBS)

DEFINES += GPU
DEFINES += OPENCV
DEFINES += DAT
# lib zbar
CONFIG+=link_pkgconfig
PKGCONFIG+=zbar
SOURCES += \
    src/Cpp/Utils/Zbar/ZbarLibs.cpp
HEADERS += \
    src/Cpp/Utils/Zbar/ZbarLibs.h
HEADERS += \
    src/Cpp/Payload/GPUBased/VDisplay.h \
    src/Cpp/Payload/GPUBased/VDisplayWorker.h \
    src/Cpp/Payload/GPUBased/VFrameGrabber.h \
    src/Cpp/Payload/GPUBased/VPreprocess.h \
    src/Cpp/Payload/GPUBased/VTrackWorker.h \
    src/Cpp/Payload/GPUBased/VODWorker.h \
    src/Cpp/Payload/GPUBased/Cuda/ip_utils.h \
    src/Cpp/Payload/GPUBased/Cuda/ipcuda_image.h \
    src/Cpp/Payload/GPUBased/Clicktrack/clicktrack.h \
    src/Cpp/Payload/GPUBased/Clicktrack/platedetector.h \
    src/Cpp/Payload/GPUBased/Clicktrack/preprocessing.h \
    src/Cpp/Payload/GPUBased/Clicktrack/recognition.h \
    src/Cpp/Payload/GPUBased/Multitrack/Dtracker.h \
    src/Cpp/Payload/GPUBased/Multitrack/Hungarian.h \
    src/Cpp/Payload/GPUBased/Multitrack/multitrack.h \
    src/Cpp/Payload/GPUBased/VMOTWorker.h \
    src/Cpp/Payload/GPUBased/VSearchWorker.h \
    src/Cpp/Payload/GPUBased/plateOCR/PlateOCR.h \
    src/Cpp/Payload/GPUBased/plateOCR/PlateOCR.h \
    src/Cpp/Payload/GPUBased/OD/yolo_v2_class.hpp

SOURCES += \
    src/Cpp/Payload/GPUBased/Clicktrack/clicktrack.cpp \
    src/Cpp/Payload/GPUBased/Clicktrack/platedetector.cpp \
    src/Cpp/Payload/GPUBased/Clicktrack/preprocessing.cpp \
    src/Cpp/Payload/GPUBased/Clicktrack/recognition.cpp \
    src/Cpp/Payload/GPUBased/Multitrack/Dtracker.cpp \
    src/Cpp/Payload/GPUBased/Multitrack/Hungarian.cpp \
    src/Cpp/Payload/GPUBased/Multitrack/multitrack.cpp \
    src/Cpp/Payload/GPUBased/VMOTWorker.cpp \
    src/Cpp/Payload/GPUBased/VSearchWorker.cpp \
    src/Cpp/Payload/GPUBased/plateOCR/PlateOCR.cpp \
    src/Cpp/Payload/GPUBased/VDisplay.cpp \
    src/Cpp/Payload/GPUBased/VDisplayWorker.cpp \
    src/Cpp/Payload/GPUBased/VFrameGrabber.cpp \
    src/Cpp/Payload/GPUBased/VPreprocess.cpp \
    src/Cpp/Payload/GPUBased/VTrackWorker.cpp \
    src/Cpp/Payload/GPUBased/VODWorker.cpp \
    src/Cpp/Payload/GPUBased/Cuda/ip_utils.cpp
DISTFILES += \
    src/Cpp/Payload/GPUBased/Cuda/ipcuda_image.cu \
    src/Cpp/Payload/GPUBased/OD/yolo-setup/yolov3-tiny_3l_last.weights \
    src/Cpp/Payload/GPUBased/OD/yolo-setup/yolov3-tiny_best.weights \
    src/Cpp/Payload/GPUBased/OD/libdarknet.so \
    src/Cpp/Payload/GPUBased/OD/yolo-setup/visdrone2019.names \
    src/Cpp/Payload/GPUBased/OD/yolo-setup/yolov3-tiny.cfg \
    src/Cpp/Payload/GPUBased/OD/yolo-setup/yolov3-tiny_3l.cfg \
    src/Cpp/Payload/GPUBased/Clicktrack/yolo-setup/yolov3-tiny_best.weights \
    src/Cpp/Payload/GPUBased/Clicktrack/mynet.pb \
    src/Cpp/Payload/GPUBased/Clicktrack/yolo-setup/yolov3-tiny.cfg
HEADERS += \
    src/Cpp/Payload/Algorithms/search/ipsearch_oppoColor.h \
    src/Cpp/Payload/Algorithms/search/ipsearch_orbextractor.h \
    src/Cpp/Payload/Algorithms/search/ipsearch_orbSearcher.h \
    src/Cpp/Payload/Algorithms/search/ipsearch_stats.h \
    src/Cpp/Payload/Algorithms/search/ipsearch_utils.h
SOURCES += \
    src/Cpp/Payload/Algorithms/search/featureColorName.cpp \
    src/Cpp/Payload/Algorithms/search/ipsearch_colornames.cpp \
    src/Cpp/Payload/Algorithms/search/ipsearch_oppoColor.cpp \
    src/Cpp/Payload/Algorithms/search/ipsearch_orbextractor.cpp \
    src/Cpp/Payload/Algorithms/search/ipsearch_orbSearcher.cpp \
    src/Cpp/Payload/Algorithms/search/ipsearch_utils.cpp
}

# Camera libs
use_video_cpu{
DEFINES += USE_VIDEO_CPU
    unix:!macx: LIBS += -L/usr/local/lib/  \
        -lopencv_objdetect \
        -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_features2d -lopencv_calib3d \
        -lopencv_videostab \
        -lopencv_video \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_videoio

    unix:!macx: DEPENDPATH += /usr/local/include
    unix:!macx: INCLUDEPATH += /usr/include/gstreamer-1.0
    unix:!macx: INCLUDEPATH += /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include
    unix:!macx: INCLUDEPATH += /usr/include/glib-2.0
    unix:!macx: INCLUDEPATH += /usr/lib/x86_64-linux-gnu/glib-2.0/include

    unix:!macx: LIBS += -LD:\usr\lib\x86_64-linux-gnu\
        -lglib-2.0 \
        -lgstreamer-1.0 \
        -lgstapp-1.0 \
        -lgstrtsp-1.0 \
        -lgstrtspserver-1.0 \
        -lgobject-2.0 \
        -lgstvideo-1.0
    unix:!macx: INCLUDEPATH += /usr/local/include
    unix:!macx: DEPENDPATH += /usr/local/include
SOURCES += \
    src/Cpp/Payload/CPUBased/CVVideoCapture.cpp \
    src/Cpp/Payload/CPUBased/CVVideoCaptureThread.cpp \
    src/Cpp/Payload/CPUBased/CVVideoProcess.cpp
HEADERS += \
    src/Cpp/Payload/CPUBased/CVVideoCapture.h \
    src/Cpp/Payload/CPUBased/CVVideoCaptureThread.h \
    src/Cpp/Payload/CPUBased/CVVideoProcess.h \
    src/Cpp/Payload/CPUBased/gstreamer_element.h
}

use_line_detector{
DEFINES += USE_LINE_DETECTOR
# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target

#============= myUnilitis lib
INCLUDEPATH += $$HOME/power_line_inspecter/power_line_inspecter/src
LIBS += -L$$HOME/power_line_inspecter/power_line_inspecter/src/build -lpli_lib
}


