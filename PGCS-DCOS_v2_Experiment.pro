QT += qml quick webengine multimedia network positioning sensors core gui serialport
QT += quickcontrols2
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

CONFIG += use_flight_control

#CONFIG += use_ucapi

#CONFIG += use_camera_control

#CONFIG += use_video_gpu

#CONFIG += use_video_cpu

QML_IMPORT_PATH += \
    $$PWD \
    $$PWD/src/QmlControls \
    $$PWD/src/Controller
SOURCES += \
    main.cpp \
    src/Camera/VideoDisplay/ImageItem.cpp

# Flight controller
use_flight_control{
INCLUDEPATH += src/libs/mavlink/include/mavlink/v2.0
INCLUDEPATH += src/libs/mavlink/include/mavlink/v2.0/ardupilotmega

SOURCES += \
    src/Controller/Com/IOFlightController.cpp \
    src/Controller/Com/LinkInterface.cpp \
    src/Controller/Com/LinkInterfaceManager.cpp \
    src/Controller/Com/MessageManager.cpp \
    src/Controller/Com/QGCMAVLink.cc \
    src/Controller/Com/TCPLink.cpp \
    src/Controller/Com/SerialLink.cpp \
    src/Controller/Com/UDPMulticastLink.cpp \
    src/Controller/Com/UDPLink.cpp \
    src/Controller/Com/ProtocolInterface.cpp \
    src/Controller/Com/MavlinkProtocol.cpp \
    src/Controller/Firmware/APM/ArduCopterFirmware.cpp \
    src/Controller/Firmware/APM/QuadPlaneFirmware.cpp \
    src/Controller/Firmware/FirmwarePlugin.cpp \
    src/Controller/Firmware/FirmwarePluginManager.cpp \
    src/Controller/Mission/MissionController.cpp \
    src/Controller/Mission/MissionItem.cpp \
    src/Controller/Mission/PlanController.cpp \
    src/Controller/Vehicle/Vehicle.cpp \
    src/Files/FileControler.cpp \
    src/Files/PlateLog.cpp \
    src/Files/PlateLogThread.cpp \
    src/Joystick/JoystickLib/JoystickController.cpp \
    src/Joystick/JoystickLib/JoystickTask.cpp \
    src/Joystick/JoystickLib/JoystickThreaded.cpp \
    src/Controller/Params/ParamsController.cpp \
    src/Setting/pcs.cpp \
    src/Setting/tinyxml2.cpp \
    src/Setting/fcs.cpp \
    src/Setting/uc.cpp \
    src/Setting/config.cpp \
    src/Maplib/Marker/Marker.cpp \
    src/Maplib/Marker/MarkerList.cpp \
    src/Maplib/Elevation.cpp \
    src/Controller/UAS/UAS.cpp \
    src/Controller/UAS/UASMessage.cpp \
    src/Machine/computer.cpp \
    src/Machine/ConnectionChecking.cpp \
    src/Machine/ConnectionThread.cpp \
    src/Maplib/profilepath.cpp \
    src/Controller/Params/Fact.cpp \
    src/Log/LogController.cpp

HEADERS += \
    src/Controller/Com/IOFlightController.h \
    src/Controller/Com/LinkInterface.h \
    src/Controller/Com/LinkInterfaceManager.h \
    src/Controller/Com/MessageManager.h \
    src/Controller/Com/QGCMAVLink.h \
    src/Controller/Com/TCPLink.h \
    src/Controller/Com/SerialLink.h \
    src/Controller/Com/UDPMulticastLink.h \
    src/Controller/Com/UDPLink.h \
    src/Controller/Com/ProtocolInterface.h \
    src/Controller/Com/MavlinkProtocol.h \
    src/Controller/Firmware/APM/ArduCopterFirmware.h \
    src/Controller/Firmware/APM/QuadPlaneFirmware.h \
    src/Controller/Firmware/FirmwarePlugin.h \
    src/Controller/Firmware/FirmwarePluginManager.h \
    src/Controller/Mission/MissionController.h \
    src/Controller/Mission/MissionItem.h \
    src/Controller/Mission/PlanController.h \
    src/Controller/Vehicle/Vehicle.h \
    src/Files/FileControler.h \
    src/Files/PlateLog.h \
    src/Files/PlateLogThread.h \
    src/Joystick/JoystickLib/JoystickController.h \
    src/Joystick/JoystickLib/JoystickTask.h \
    src/Joystick/JoystickLib/JoystickThreaded.h \
    src/Controller/Params/ParamsController.h \
    src/Setting/pcs.h \
    src/Setting/tinyxml2.h \
    src/Setting/fcs.h \
    src/Setting/uc.h \
    src/Setting/config.h \
    src/Maplib/Marker/Marker.h \
    src/Maplib/Marker/MarkerList.h \
    src/Maplib/Elevation.h \
    src/Controller/UAS/UAS.h \
    src/Controller/UAS/UASMessage.h \
    src/Machine/computer.hpp \
    src/Machine/ConnectionChecking.h \
    src/Machine/ConnectionThread.h \
    src/Maplib/profilepath.h \
    src/Controller/Params/Fact.h \
    src/Log/LogController.h
}
# UC libs KURENTO
use_ucapi{
DEFINES += UC_API
DEFINES += SIO_TLS
QML_IMPORT_PATH += \
    $$PWD/src/QmlControls/UC
INCLUDEPATH += src/UC
SOURCES += \
    src/UC/UCDataModel.cpp \
    src/UC/UCEventListener.cpp
HEADERS += \
    src/UC/UCDataModel.hpp \
    src/UC/UCEventListener.hpp

INCLUDEPATH += $$PWD/src/UC/sioclient/lib
INCLUDEPATH += $$PWD/src/UC/boost1.62/include
INCLUDEPATH += $$PWD/src/UC/openssl

#LIBS += -L$$PWD/src/UC/lib -lboost_system -lboost_chrono -lboost_thread -lboost_timer
LIBS+= -L/home/pgcs-05/workspaces/boost1.62/lib -lboost_system -lboost_chrono -lboost_thread -lboost_timer
LIBS += -L/usr/local/lib -lcrypto -lssl
LIBS += -lpthread

SOURCES += \
    src/UC/api/app_socket_api.cpp \
    src/UC/json/jsoncpp.cpp \
    src/UC/sioclient/src/internal/sio_client_impl.cpp \
    src/UC/sioclient/src/internal/sio_packet.cpp \
    src/UC/sioclient/src/sio_client.cpp \
    src/UC/sioclient/src/sio_socket.cpp

HEADERS += \
    src/UC/api/app_socket_api.hpp \
    src/UC/json/json-forwards.h \
    src/UC/json/json.h \
    src/UC/sioclient/src/internal/sio_client_impl.h \
    src/UC/sioclient/src/internal/sio_packet.h \
    src/UC/sioclient/src/sio_client.h \
    src/UC/sioclient/src/sio_message.h \
    src/UC/sioclient/src/sio_socket.h

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
    src/Camera/ControllerLib/Buffer/BufferOut.cpp \
    src/Camera/ControllerLib/Command/GeoCommands.cpp \
    src/Camera/ControllerLib/Command/IPCCommands.cpp \
    src/Camera/ControllerLib/Command/MotionCCommands.cpp \
    src/Camera/ControllerLib/Command/SystemCommands.cpp \
    src/Camera/ControllerLib/ByteManipulation.cpp \
    src/Camera/ControllerLib/georeferencecontext.cpp \
    src/Camera/ControllerLib/gimbalcontext.cpp \
    src/Camera/ControllerLib/gimbalinterfacecontext.cpp \
    src/Camera/ControllerLib/gimbalpacketparser.cpp \
    src/Camera/ControllerLib/networkparameterscontext.cpp \
    src/Camera/ControllerLib/samplegimbal.cpp \
    src/Camera/ControllerLib/udppayload.cpp \
    src/Camera/ControllerLib/versioncontext.cpp \
    src/Camera/ControllerLib/tcp/clientStuff.cpp \
    src/Camera/ControllerLib/tcp/gimbal_control.cpp \
    src/Camera/ControllerLib/EPTools/EPHucomTool.cpp \
    src/Camera/Cache/Cache.cpp

HEADERS += \
    src/Camera/ControllerLib/Buffer/BufferOut.h \
    src/Camera/ControllerLib/Buffer/RollBuffer.h \
    src/Camera/ControllerLib/Buffer/RollBuffer_.h \
    src/Camera/ControllerLib/Buffer/RollBuffer_q.h \
    src/Camera/ControllerLib/Command/GeoCommands.h \
    src/Camera/ControllerLib/Command/IPCCommands.h \
    src/Camera/ControllerLib/Command/MotionCCommands.h \
    src/Camera/ControllerLib/Command/SystemCommands.h \
    src/Camera/ControllerLib/Packet/Common_type.h \
    src/Camera/ControllerLib/Packet/Confirm.h \
    src/Camera/ControllerLib/Packet/EnGeoLocation.h \
    src/Camera/ControllerLib/Packet/EOS.h \
    src/Camera/ControllerLib/Packet/EyeCheck.h \
    src/Camera/ControllerLib/Packet/EyeEvent.h \
    src/Camera/ControllerLib/Packet/EyephoenixProtocol.h \
    src/Camera/ControllerLib/Packet/EyeRotationMatrix.h \
    src/Camera/ControllerLib/Packet/EyeStatus.h \
    src/Camera/ControllerLib/Packet/GimbalMode.h \
    src/Camera/ControllerLib/Packet/GimbalRecord.h \
    src/Camera/ControllerLib/Packet/GimbalRecordStatus.h \
    src/Camera/ControllerLib/Packet/GimbalStab.h \
    src/Camera/ControllerLib/Packet/GPSData.h \
    src/Camera/ControllerLib/Packet/GPSRate.h \
    src/Camera/ControllerLib/Packet/HConfigMessage.h \
    src/Camera/ControllerLib/Packet/ImageStab.h \
    src/Camera/ControllerLib/Packet/InstallMode.h \
    src/Camera/ControllerLib/Packet/IPCStatusResponse.h \
    src/Camera/ControllerLib/Packet/KLV.h \
    src/Camera/ControllerLib/Packet/LockMode.h \
    src/Camera/ControllerLib/Packet/Matrix.h \
    src/Camera/ControllerLib/Packet/MCParams.h \
    src/Camera/ControllerLib/Packet/MData.h \
    src/Camera/ControllerLib/Packet/MotionAngle.h \
    src/Camera/ControllerLib/Packet/MotionCStatus.h \
    src/Camera/ControllerLib/Packet/MotionImage.h \
    src/Camera/ControllerLib/Packet/Object.h \
    src/Camera/ControllerLib/Packet/PTAngle.h \
    src/Camera/ControllerLib/Packet/PTAngleDiff.h \
    src/Camera/ControllerLib/Packet/PTRateFactor.h \
    src/Camera/ControllerLib/Packet/RapidView.h \
    src/Camera/ControllerLib/Packet/RequestResponsePacket.h \
    src/Camera/ControllerLib/Packet/RFData.h \
    src/Camera/ControllerLib/Packet/RFRequest.h \
    src/Camera/ControllerLib/Packet/RTData.h \
    src/Camera/ControllerLib/Packet/SceneSteering.h \
    src/Camera/ControllerLib/Packet/ScreenPoint.h \
    src/Camera/ControllerLib/Packet/SensorColor.h \
    src/Camera/ControllerLib/Packet/SensorId.h \
    src/Camera/ControllerLib/Packet/Snapshot.h \
    src/Camera/ControllerLib/Packet/StreamingProfile.h \
    src/Camera/ControllerLib/Packet/SystemStatus.h \
    src/Camera/ControllerLib/Packet/TargetPosition.h \
    src/Camera/ControllerLib/Packet/Telemetry.h \
    src/Camera/ControllerLib/Packet/TrackObject.h \
    src/Camera/ControllerLib/Packet/TrackResponse.h \
    src/Camera/ControllerLib/Packet/TrackSize.h \
    src/Camera/ControllerLib/Packet/utils.h \
    src/Camera/ControllerLib/Packet/Vector.h \
    src/Camera/ControllerLib/Packet/XPoint.h \
    src/Camera/ControllerLib/Packet/ZoomData.h \
    src/Camera/ControllerLib/Packet/ZoomStatus.h \
    src/Camera/ControllerLib/ByteManipulation.h \
    src/Camera/ControllerLib/georeferencecontext.h \
    src/Camera/ControllerLib/gimbalcontext.h \
    src/Camera/ControllerLib/gimbalinterfacecontext.h \
    src/Camera/ControllerLib/gimbalpacketparser.h \
    src/Camera/ControllerLib/ipendpoint.h \
    src/Camera/ControllerLib/networkparameterscontext.h \
    src/Camera/ControllerLib/samplegimbal.h \
    src/Camera/ControllerLib/UavvGimbalProtocol.h \
    src/Camera/ControllerLib/udppayload.h \
    src/Camera/ControllerLib/versioncontext.h \
    src/Camera/Cache/Cache.h \
    src/Camera/Cache/TrackObject.h \
    src/Camera/Cache/CacheItem.h \
    src/Camera/Cache/DetectedObjectsCacheItem.h \
    src/Camera/Cache/FixedMemory.h \
    src/Camera/Cache/GstFrameCacheItem.h \
    src/Camera/Cache/ProcessImageCacheItem.h \
    src/Camera/ControllerLib/tcp/clientStuff.h \
    src/Camera/ControllerLib/tcp/gimbal_control.h \
    src/Camera/ControllerLib/EPTools/EPHucomTool.h \
    src/Camera/ControllerLib/EPTools/EPSensorTool.h
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
CUDA_SOURCES += src/Camera/GPUBased/Cuda/ipcuda_image.cu

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
INCLUDEPATH += /home/pgcs-05/install/tensorflow/tensorflow-1.14.0
INCLUDEPATH += /home/pgcs-05/install/tensorflow/tensorflow-1.14.0/tensorflow
INCLUDEPATH += /home/pgcs-05/install/tensorflow/tensorflow-1.14.0/bazel-tensorflow-1.14.0/external/eigen_archive
INCLUDEPATH += /home/pgcs-05/install/tensorflow/tensorflow-1.14.0/bazel-tensorflow-1.14.0/external/protobuf_archive/src
INCLUDEPATH += /home/pgcs-05/install/tensorflow/tensorflow-1.14.0/bazel-genfiles

LIBS += -L/home/pgcs-05/install/tensorflow/tensorflow-1.14.0/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework

LIBS += `pkg-config --libs opencv`
# End TensorFlow
#LIBS += -LD:\usr\local\lib \
# -ldarknet

#LIBS += -L/home/pgcs-05/Downloads/darknet-GPU_blob_click_updateLayers_I420 -ldarknet
LIBS += /home/pgcs-05/workspaces/darknet-GPU_blob_click_updateLayers_I420/libdarknet.so
message($$LIBS)

DEFINES += GPU
DEFINES += OPENCV
DEFINES += DAT
# lib zbar
CONFIG+=link_pkgconfig
PKGCONFIG+=zbar
HEADERS += \
    src/Zbar/ZbarLibs.h \
    src/Camera/GPUBased/stabilizer/dando_02/stab_gcs_kiir.hpp \
    src/Camera/GPUBased/tracker/dando/HTrack/ffttools.hpp \
    src/Camera/GPUBased/tracker/dando/HTrack/fhog.hpp \
    src/Camera/GPUBased/tracker/dando/HTrack/saliency.h \
    src/Camera/GPUBased/tracker/dando/LME/lme.hpp \
    src/Camera/GPUBased/tracker/dando/ITrack.hpp \
    src/Camera/GPUBased/tracker/dando/Utilities.hpp \
    src/Camera/GPUBased/tracker/mosse/tracker.h \
    src/Camera/GPUBased/VDisplay.h \
    src/Camera/GPUBased/VDisplayWorker.h \
    src/Camera/GPUBased/VFrameGrabber.h \
    src/Camera/GPUBased/VPreprocess.h \
    src/Camera/GPUBased/VTrackWorker.h \
    src/Camera/GPUBased/VODWorker.h \
    src/Camera/GPUBased/Cuda/ip_utils.h \
    src/Camera/GPUBased/Cuda/ipcuda_image.h \
    src/Camera/GPUBased/VRTSPServer.h \
    src/Camera/GPUBased/VSavingWorker.h \
    src/Camera/GPUBased/Clicktrack/clicktrack.h \
    src/Camera/GPUBased/Clicktrack/platedetector.h \
    src/Camera/GPUBased/Clicktrack/preprocessing.h \
    src/Camera/GPUBased/Clicktrack/recognition.h \
    src/Camera/GPUBased/Multitrack/Dtracker.h \
    src/Camera/GPUBased/Multitrack/Hungarian.h \
    src/Camera/GPUBased/Multitrack/multitrack.h \
    src/Camera/GPUBased/VMOTWorker.h \
    src/Camera/GPUBased/VSearchWorker.h \
    src/Camera/GPUBased/plateOCR/PlateOCR.h \
    src/Camera/GPUBased/plateOCR/PlateOCR.h \
    src/Camera/GPUBased/OD/yolo_v2_class.hpp

SOURCES += \
    src/Zbar/ZbarLibs.cpp \
    src/Camera/GPUBased/stabilizer/dando_02/stab_gcs_kiir.cpp \
    src/Camera/GPUBased/tracker/dando/HTrack/ffttools.cpp \
    src/Camera/GPUBased/tracker/dando/HTrack/fhog.cpp \
    src/Camera/GPUBased/tracker/dando/HTrack/saliency.cpp \
    src/Camera/GPUBased/tracker/dando/LME/lme.cpp \
    src/Camera/GPUBased/tracker/dando/ITrack.cpp \
    src/Camera/GPUBased/tracker/dando/Utilities.cpp \
    src/Camera/GPUBased/tracker/mosse/tracker.cpp \
    src/Camera/GPUBased/Clicktrack/clicktrack.cpp \
    src/Camera/GPUBased/Clicktrack/platedetector.cpp \
    src/Camera/GPUBased/Clicktrack/preprocessing.cpp \
    src/Camera/GPUBased/Clicktrack/recognition.cpp \
    src/Camera/GPUBased/Multitrack/Dtracker.cpp \
    src/Camera/GPUBased/Multitrack/Hungarian.cpp \
    src/Camera/GPUBased/Multitrack/multitrack.cpp \
    src/Camera/GPUBased/VMOTWorker.cpp \
    src/Camera/GPUBased/VSearchWorker.cpp \
    src/Camera/GPUBased/plateOCR/PlateOCR.cpp \
    src/Camera/GPUBased/VDisplay.cpp \
    src/Camera/GPUBased/VDisplayWorker.cpp \
    src/Camera/GPUBased/VFrameGrabber.cpp \
    src/Camera/GPUBased/VPreprocess.cpp \
    src/Camera/GPUBased/VTrackWorker.cpp \
    src/Camera/GPUBased/VODWorker.cpp \
    src/Camera/GPUBased/Cuda/ip_utils.cpp \
    src/Camera/GPUBased/VSavingWorker.cpp \
    src/Camera/GPUBased/VRTSPServer.cpp
DISTFILES += \
    src/Camera/GPUBased/Cuda/ipcuda_image.cu \
    src/Camera/GPUBased/OD/yolo-setup/yolov3-tiny_3l_last.weights \
    src/Camera/GPUBased/OD/yolo-setup/yolov3-tiny_best.weights \
    src/Camera/GPUBased/OD/libdarknet.so \
    src/Camera/GPUBased/OD/yolo-setup/visdrone2019.names \
    src/Camera/GPUBased/OD/yolo-setup/yolov3-tiny.cfg \
    src/Camera/GPUBased/OD/yolo-setup/yolov3-tiny_3l.cfg \
    src/Camera/GPUBased/Clicktrack/yolo-setup/yolov3-tiny_best.weights \
    src/Camera/GPUBased/Clicktrack/mynet.pb \
    src/Camera/GPUBased/Clicktrack/yolo-setup/yolov3-tiny.cfg
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
    src/Camera/CPUBased/convert/VideoConverter.cpp \
    src/Camera/CPUBased/convert/VideoConverterThread.cpp \
    src/Camera/CPUBased/copy/FileCopy.cpp \
    src/Camera/CPUBased/copy/filecopythread.cpp \
    src/Camera/CPUBased/detector/movingObject.cpp \
    src/Camera/CPUBased/recorder/gstsaver.cpp \
    src/Camera/CPUBased/recorder/gstutility.cpp \
    src/Camera/CPUBased/stream/CVRecord.cpp \
    src/Camera/CPUBased/stream/CVVideoCapture.cpp \
    src/Camera/CPUBased/stream/CVVideoCaptureThread.cpp \
    src/Camera/CPUBased/stream/CVVideoProcess.cpp \
    src/Camera/CPUBased/stream/VRTSPServer.cpp \
    src/Camera/CPUBased/stream/VSavingWorker.cpp \
    src/Camera/CPUBased/stabilizer/dando_02/stab_gcs_kiir.cpp \
    src/Camera/CPUBased/tracker/dando/LME/lme.cpp \
    src/Camera/CPUBased/tracker/dando/HTrack/ffttools.cpp \
    src/Camera/CPUBased/tracker/dando/HTrack/fhog.cpp \
    src/Camera/CPUBased/tracker/dando/HTrack/saliency.cpp \
    src/Camera/CPUBased/tracker/dando/SKCF/gradient.cpp \
    src/Camera/CPUBased/tracker/dando/SKCF/skcf.cpp \
    src/Camera/CPUBased/tracker/dando/ITrack.cpp \
    src/Camera/CPUBased/tracker/dando/Utilities.cpp \
    src/Camera/CPUBased/tracker/dando/movdetection/kltwrapper.cpp \
    src/Camera/CPUBased/tracker/dando/movdetection/mcdwrapper.cpp \
    src/Camera/CPUBased/tracker/dando/InitTracking.cpp \
    src/Camera/CPUBased/tracker/dando/kalman.cpp \
    src/Camera/CPUBased/tracker/dando/thresholding.cpp \
    src/Camera/CPUBased/tracker/dando/image_utils.cpp
HEADERS += \
    src/Camera/CPUBased/convert/VideoConverter.h \
    src/Camera/CPUBased/convert/VideoConverterThread.h \
    src/Camera/CPUBased/copy/FileCopy.h \
    src/Camera/CPUBased/copy/filecopythread.h \
    src/Camera/CPUBased/detector/movingObject.hpp \
    src/Camera/CPUBased/packet/Common_type.h \
    src/Camera/CPUBased/packet/EyephoenixProtocol.h \
    src/Camera/CPUBased/packet/KLV.h \
    src/Camera/CPUBased/packet/Matrix.h \
    src/Camera/CPUBased/packet/MData.h \
    src/Camera/CPUBased/packet/MotionImage.h \
    src/Camera/CPUBased/packet/Object.h \
    src/Camera/CPUBased/packet/utils.h \
    src/Camera/CPUBased/packet/Vector.h \
    src/Camera/CPUBased/recorder/gstsaver.hpp \
    src/Camera/CPUBased/recorder/gstutility.hpp \
    src/Camera/CPUBased/stream/CVRecord.h \
    src/Camera/CPUBased/stream/CVVideoCapture.h \
    src/Camera/CPUBased/stream/CVVideoCaptureThread.h \
    src/Camera/CPUBased/stream/CVVideoProcess.h \
    src/Camera/CPUBased/stream/gstreamer_element.h \
    src/Camera/CPUBased/stream/VRTSPServer.h \
    src/Camera/CPUBased/stream/VSavingWorker.h \
    src/Camera/CPUBased/stabilizer/dando_02/stab_gcs_kiir.hpp \
    src/Camera/CPUBased/tracker/dando/SKCF/gradient.h \
    src/Camera/CPUBased/tracker/dando/SKCF/skcf.h \
    src/Camera/CPUBased/tracker/dando/ITrack.hpp \
    src/Camera/CPUBased/tracker/dando/Utilities.hpp \
    src/Camera/CPUBased/tracker/dando/ktracker.h \
    src/Camera/CPUBased/tracker/dando/movdetection/kltwrapper.hpp \
    src/Camera/CPUBased/tracker/dando/movdetection/mcdwrapper.hpp \
    src/Camera/CPUBased/tracker/dando/movdetection/params.hpp \
    src/Camera/CPUBased/tracker/dando/movdetection/prob_model.hpp \
    src/Camera/CPUBased/tracker/dando/InitTracking.hpp \
    src/Camera/CPUBased/tracker/dando/kalman.hpp \
    src/Camera/CPUBased/tracker/dando/thresholding.hpp \
    src/Camera/CPUBased/tracker/dando/image_utils.hpp \
    src/Camera/CPUBased/tracker/dando/LME/lme.hpp \
    src/Camera/CPUBased/tracker/dando/HTrack/ffttools.hpp \
    src/Camera/CPUBased/tracker/dando/HTrack/fhog.hpp \
    src/Camera/CPUBased/tracker/dando/HTrack/saliency.h \
    src/Camera/CPUBased/tracker/dando/ITrack.hpp \
    src/Camera/CPUBased/tracker/dando/Utilities.hpp
}

HEADERS += \
    src/Camera/VideoDisplay/ImageItem.h

