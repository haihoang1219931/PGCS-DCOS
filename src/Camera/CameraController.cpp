#include "CameraController.h"
#ifdef USE_VIDEO_CPU
#include "Camera/CPUBased/stream/CVVideoCaptureThread.h"
#elif USE_VIDEO_GPU
#include "Camera/GPUBased/VDisplay.h"
#endif
CameraController::CameraController(QObject *parent) : QObject(parent)
{
#ifdef USE_VIDEO_GPU
    m_videoEngine = new VDisplay();
#elif USE_VIDEO_CPU
    m_videoEngine = new CVVideoCaptureThread();
#endif
    m_gimbalManager = new GimbalInterfaceManager();

    //    m_videoEngine->decoder()->setContext(m_gimbal->context());
}
void CameraController::loadConfig(Config *config){
    if(config != nullptr){
        m_config = config;
        QString gimbalType = m_config->value("Settings:GimbalType:Value:data").toString();
        if(gimbalType == "CM160"){
            m_gimbal = m_gimbalManager->getGimbal(GimbalInterfaceManager::GIMBAL_TYPE::CM160);
        }else if(gimbalType == "GREMSY"){
            m_gimbal = m_gimbalManager->getGimbal(GimbalInterfaceManager::GIMBAL_TYPE::GREMSY);
        }else if(gimbalType == "TRERON"){
            m_gimbal = m_gimbalManager->getGimbal(GimbalInterfaceManager::GIMBAL_TYPE::TRERON);
        }else if(gimbalType == "SBUS"){
            m_gimbal = m_gimbalManager->getGimbal(GimbalInterfaceManager::GIMBAL_TYPE::SBUS);
        }else{
            m_gimbal = m_gimbalManager->getGimbal(GimbalInterfaceManager::GIMBAL_TYPE::UNKNOWN);
        }
        m_gimbal->setVideoEngine(m_videoEngine);
        m_videoEngine->setGimbal(m_gimbal);
        m_gimbal->connectToGimbal(config);
        m_videoEngine->loadConfig(config);
    }
}
CameraController::~CameraController(){
    if(m_gimbal != nullptr){
        m_gimbal->deleteLater();
    }
    if(m_videoEngine != nullptr){
        m_videoEngine->stop();
        m_videoEngine->deleteLater();
    }
    printf("Delete CameraController\r\n");
}
