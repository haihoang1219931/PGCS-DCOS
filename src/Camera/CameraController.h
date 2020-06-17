#ifndef CAMERACONTROLLER_H
#define CAMERACONTROLLER_H

#include <QObject>
#include <stdio.h>
#include <unistd.h>
#include "GimbalController/GimbalInterfaceManager.h"
#include "GimbalController/GimbalData.h"
#include "VideoEngine/VideoEngineInterface.h"
#include "Setting/config.h"
#include "../Controller/Vehicle/Vehicle.h"

class CameraController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(GimbalInterface* gimbal      READ gimbal             NOTIFY gimbalChanged)
    Q_PROPERTY(GimbalData* context          READ context             NOTIFY contextChanged)
    Q_PROPERTY(VideoEngine* videoEngine     READ videoEngine NOTIFY videoEngineChanged)

public:
    explicit CameraController(QObject *parent = nullptr);    
    ~CameraController();
    GimbalInterface* gimbal(){ return m_gimbal;}
    GimbalData* context(){
        if(m_gimbal!=nullptr)
            return m_gimbal->context();
        else
            return nullptr;
    }
    VideoEngine* videoEngine(){ return m_videoEngine; }


    Q_INVOKABLE void loadConfig(Config *config);

Q_SIGNALS:
    void gimbalChanged();
    void contextChanged();
    void videoEngineChanged();
private:
    Config* m_config;
    GimbalInterfaceManager* m_gimbalManager = nullptr;
    GimbalInterface* m_gimbal = nullptr;
    VideoEngine* m_videoEngine = nullptr;

};

#endif // CAMERACONTROLLER_H
