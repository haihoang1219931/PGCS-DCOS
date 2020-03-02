#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtWebEngine/QtWebEngine>
//--- Flight controller
#include "src/Controller/Com/IOFlightController.h"
#include "src/Controller/Vehicle/Vehicle.h"
#include "src/Controller/Mission/PlanController.h"
#include "src/Controller/Mission/MissionController.h"
#include "src/Controller/Params/ParamsController.h"
#include "src/Maplib/Elevation.h"
#include "src/Maplib/Marker/MarkerList.h"
#include "src/Machine/computer.hpp"
//----Model
#include "src/Maplib/profilepath.h"
#ifdef CAMERA_CONTROL
    #include "src/Camera/ControllerLib/samplegimbal.h"
#endif
//--- UC
#ifdef UC_API
    #include "api/app_socket_api.hpp"
    #include "src/UC/UCEventListener.hpp"
    #include "src/UC/UCDataModel.hpp"
#endif
//--- Payload controller
#ifdef VIDEO_DECODER
    #include "src/Camera/CPUBased/stream/CVVideoCaptureThread.h"
#endif
//--- GPU Process
#ifdef GPU_PROCESS
    #include "src/Camera/GPUBased/Video/VDisplay.h"
    #include "src/Camera/Cache/Cache.h"
    Cache *Cache::m_instance = nullptr;
#endif

//--- Joystick
#include "src/Joystick/JoystickLib/JoystickThreaded.h"

//--- Config
#include "src/Setting/fcs.h"
#include "src/Setting/uc.h"
#include "src/Setting/pcs.h"
int main(int argc, char *argv[])
{

    QGuiApplication app(argc, argv);
    app.setOrganizationName("qdt");
    app.setOrganizationDomain("qdt");
    QtWebEngine::initialize();
    QQmlApplicationEngine engine;
    engine.addImportPath("qrc:/");
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
    //----Model-----------
    qmlRegisterType<ProfilePath>("io.qdt.dev", 1, 0,    "ProfilePath");
//    qmlRegisterType<MAV_TYPE>("io.qdt.dev", 1, 0, "MAV_TYPE", "MAV_TYPE");
    FCSConfig fcsConfig;
    fcsConfig.readConfig(QGuiApplication::applicationDirPath() + "/conf/fcs.conf");
    engine.rootContext()->setContextProperty("FCSConfig", &fcsConfig);
    FCSConfig trkConfig;
    trkConfig.readConfig(QGuiApplication::applicationDirPath() + "/conf/trk.conf");
    engine.rootContext()->setContextProperty("TRKConfig", &trkConfig);
#ifdef VIDEO_DECODER
    //--- Camera controller
    qmlRegisterType<CVVideoCaptureThread>("io.qdt.dev", 1, 0, "Player");
    engine.rootContext()->setContextProperty("VIDEO_DECODER", QVariant(true));
#else
    engine.rootContext()->setContextProperty("VIDEO_DECODER", QVariant(false));
#endif
#ifdef GPU_PROCESS
    Cache::instance();
    qmlRegisterType<VDisplay>("io.qdt.dev", 1, 0, "Player");
    engine.rootContext()->setContextProperty("GPU_PROCESS", QVariant(true));
#else
    engine.rootContext()->setContextProperty("GPU_PROCESS", QVariant(false));
#endif
#ifdef CAMERA_CONTROL
    PCSConfig pcsConfig;
    pcsConfig.readConfig(QGuiApplication::applicationDirPath() + "/conf/pcs.conf");
    engine.rootContext()->setContextProperty("PCSConfig", &pcsConfig);
    qmlRegisterType<SampleGimbal>("io.qdt.dev", 1, 0, "GimbalNetwork");
    engine.rootContext()->setContextProperty("CAMERA_CONTROL", QVariant(true));
#else
    qmlRegisterType<QObject>("io.qdt.dev", 1, 0, "GimbalNetwork");
    engine.rootContext()->setContextProperty("CAMERA_CONTROL", QVariant(false));
#endif
#ifdef UC_API
    //--- UC Socket API
    UCConfig ucConfig;
    ucConfig.readConfig(QGuiApplication::applicationDirPath() + "/conf/uc.conf");
    AppSocketApi *appSocketApi = AppSocketApi::connectToServer(
                                     ucConfig.value("Settings:UCServerAddress:Value:data").toString(),
                                     ucConfig.value("Settings:UCServerPort:Value:data").toInt(),
                                     ucConfig.value("Settings:UCServerName:Value:data").toString());

    appSocketApi->createNewRoom(
        ucConfig.value("Settings:UCStreamSource:Value:data").toString(),
        ucConfig.value("Settings:UCRoomName:Value:data").toString());
//    QObject *appSocketApi = new QObject();
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
    //--- Joystick
    JoystickThreaded::expose();
    //--- Other things
    engine.rootContext()->setContextProperty("applicationDirPath", QGuiApplication::applicationDirPath());
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
*/
