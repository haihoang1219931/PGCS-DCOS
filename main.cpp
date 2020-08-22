#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtWebEngine/QtWebEngine>
//--- Flight controller
#include "Flight/Com/IOFlightController.h"
#include "Flight/Vehicle/Vehicle.h"
#include "Flight/Mission/PlanController.h"
#include "Flight/Mission/MissionController.h"
#include "Flight/Params/ParamsController.h"
#include "Utils/Files/PlateLog.h"
#include "Utils/Maplib/Elevation.h"
#include "Utils/Maplib/Marker/MarkerList.h"
#include "Utils/Machine/computer.hpp"
//----Model nhatdn1
#include "Utils/Maplib/Model/symbol.h"
#include "Utils/Maplib/Model/symbolmodel.h"
#include "Utils/Maplib/profilepath.h"
#ifdef CAMERA_CONTROL
    #include "Payload/CameraController.h"
#endif
//--- UC
#ifdef UC_API
    #include "api/app_socket_api.hpp"
    #include "src/Cpp/UC/UCEventListener.hpp"
    #include "src/Cpp/UC/UCDataModel.hpp"
#endif
//--- Payload controller
#ifdef USE_VIDEO_CPU
    #include "Payload/VideoDisplay/ImageItem.h"
    #include "Payload/CPUBased/CVVideoCaptureThread.h"
#endif
//--- GPU Process
#ifdef USE_VIDEO_GPU
    #include "src/Camera/VideoDisplay/ImageItem.h"
    #include "src/Camera/GPUBased/VDisplay.h"
#endif
//--- Joystick
#include "Utils/Joystick/JoystickThreaded.h"

//--- Config
#include "Utils/Setting/config.h"

// --- Network
#include "Utils/Network/NetworkManager.h"
int main(int argc, char *argv[])
{

    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
    QGuiApplication app(argc, argv);
    app.setOrganizationName("qdt");
    app.setOrganizationDomain("qdt");
    QQmlApplicationEngine engine;
    engine.addImportPath("qrc:/");
    //--- Load app config
    qmlRegisterType<ConfigElement>("io.qdt.dev", 1, 0,"ConfigElement");
    Config appConfig;
    appConfig.readConfig("conf/app.conf");
    engine.rootContext()->setContextProperty("ApplicationConfig", &appConfig);
    //set antialiasing nhatdn1
    QSurfaceFormat format;
    format.setSamples(8);
    QSurfaceFormat::setDefaultFormat(format);

#ifdef UC_API
    QtWebEngine::initialize();
#endif
#ifdef UC_API

    //--- UC Socket API
    Config ucConfig;
    ucConfig.readConfig("conf/uc.conf");
    AppSocketApi *appSocketApi = AppSocketApi::connectToServer(
                                     ucConfig.value("Settings:UCServerAddress:Value:data").toString(),
                                     ucConfig.value("Settings:UCServerPort:Value:data").toInt(),
                                     ucConfig.value("Settings:UCServerName:Value:data").toString());

    appSocketApi->createNewRoom(
        ucConfig.value("Settings:UCStreamSource:Value:data").toString(),
        ucConfig.value("Settings:UCRoomName:Value:data").toString());
    engine.rootContext()->setContextProperty("UcApi", appSocketApi);
    engine.rootContext()->setContextProperty("UcApiConfig", &ucConfig);
    //---  UC Data Binding
    UC::UCDataModel *ucDataModel = UC::UCDataModel::instance();
    engine.rootContext()->setContextProperty("UCDataModel", ucDataModel);
    UCEventListener *ucEventListener = UCEventListener::instance();
    engine.rootContext()->setContextProperty("UCEventListener", ucEventListener);
    //- UC Enum Attributes
    UC::UserAttribute::expose();
    UC::UserRoles::expose();
    UCEventEnums::expose();
    UC::RedoActionAfterReloadWebView::expose();
    engine.rootContext()->setContextProperty("UC_API", QVariant(true));
#else
    engine.rootContext()->setContextProperty("UC_API", QVariant(false));
#endif

    //--- Flight controller
    qmlRegisterType<MissionItem>("io.qdt.dev", 1, 0,    "MissionItem");
    qmlRegisterType<ParamsController>("io.qdt.dev", 1, 0, "ParamsController");
    qmlRegisterType<IOFlightController>("io.qdt.dev",1,0,"IOFlightController");
    qmlRegisterType<Fact>("io.qdt.dev",1,0,"Fact");
    qmlRegisterType<Vehicle>("io.qdt.dev",1,0,"Vehicle");
    qmlRegisterType<UAS>("io.qdt.dev",1,0,"UAS");
    qmlRegisterType<UASMessage>("io.qdt.dev",1,0,"UASMessage");
    qmlRegisterType<PlanController>("io.qdt.dev",1,0,"PlanController");
    qmlRegisterType<MissionController>("io.qdt.dev",1,0,"MissionController");
    qmlRegisterType<Elevation>("io.qdt.dev",1,0,"Elevation");
    qmlRegisterType<Marker>("io.qdt.dev",1,0,"Marker");
    qmlRegisterType<MarkerList>("io.qdt.dev",1,0,"MarkerList");
    qmlRegisterType<COMPUTER_INFO>("io.qdt.dev", 1, 0, "Computer");
    //----Model nhatdn1-----------
    qmlRegisterType<ProfilePath>("io.qdt.dev", 1, 0,    "ProfilePath");
    qmlRegisterType<SymbolModel>("io.qdt.dev", 1, 0,    "SymbolModel");

//    qmlRegisterType<MAV_TYPE>("io.qdt.dev", 1, 0, "MAV_TYPE", "MAV_TYPE");
    Config fcsConfig;
    fcsConfig.readConfig("conf/fcs.conf");
    engine.rootContext()->setContextProperty("FCSConfig", &fcsConfig);
    Config trkConfig;
    trkConfig.readConfig("conf/trk.conf");
    engine.rootContext()->setContextProperty("TRKConfig", &trkConfig);
#ifdef USE_VIDEO_CPU
    //--- Camera controller
    qmlRegisterType<TrackObjectInfo>("io.qdt.dev", 1, 0, "TrackObjectInfo");
    qmlRegisterType<CVVideoCaptureThread>("io.qdt.dev", 1, 0, "Player");
    engine.rootContext()->setContextProperty("USE_VIDEO_CPU", QVariant(true));
#else
    engine.rootContext()->setContextProperty("USE_VIDEO_CPU", QVariant(false));
#endif
#ifdef USE_VIDEO_GPU

    qmlRegisterType<TrackObjectInfo>("io.qdt.dev", 1, 0, "TrackObjectInfo");
    qmlRegisterType<VDisplay>("io.qdt.dev", 1, 0, "Player");
    engine.rootContext()->setContextProperty("USE_VIDEO_GPU", QVariant(true));
#else
    engine.rootContext()->setContextProperty("USE_VIDEO_GPU", QVariant(false));
#endif
#ifdef CAMERA_CONTROL
    Config pcsConfig;
    pcsConfig.readConfig("conf/pcs.conf");
    engine.rootContext()->setContextProperty("PCSConfig", &pcsConfig);
    qmlRegisterType<CameraController>("io.qdt.dev", 1, 0, "CameraController");
    qmlRegisterType<VideoRender>("io.qdt.dev", 1, 0, "VideoRender");
    engine.rootContext()->setContextProperty("CAMERA_CONTROL", QVariant(true));
#else
    qmlRegisterType<QObject>("io.qdt.dev", 1, 0, "VideoRender");
    qmlRegisterType<QObject>("io.qdt.dev", 1, 0, "CameraController");
    engine.rootContext()->setContextProperty("CAMERA_CONTROL", QVariant(false));
#endif
    qmlRegisterType<PlateLog>("io.qdt.dev", 1, 0, "PlateLog");
    //--- Joystick
    JoystickThreaded::expose();
    //--- Network
    NetworkManager::expose();
    //--- Other things

    engine.rootContext()->setContextProperty("applicationDirPath", QGuiApplication::applicationDirPath());
    engine.addImportPath("qrc:/");
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
/*
* Receive multicast
======= on normal pc
* sudo ip route add 224.0.0.0/4 dev enp2s0
* Kurento
* sudo systemctl enable kurento-media-server
* Share network
* sudo ip route replace default via 192.168.43.1 dev wlp3s0 proto static
======= on gpu pc
* sudo ip route add 224.0.0.0/4 dev enp0s31f6
* Kurento
* sudo systemctl enable kurento-media-server
* Share network
* sudo ip route replace default via 192.168.43.1 dev wlp1s0 proto static
* echo fs.inotify.max_user_watches=582222 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
* sudo sysctl fs.inotify.max_user_watches=582222 && sudo sysctl -p
* convert -background none -resize 40x40 VTOLPlane.svg VTOLPlane.png
======= install nodejs
sudo apt install build-essential apt-transport-https lsb-release ca-certificates curl
*/
