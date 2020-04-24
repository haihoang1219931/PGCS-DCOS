#include "GremseyGimbal.h"
#include "GimbalControl.h"
#include "TCPClient.h"
#include "Camera/VideoEngine/VideoEngineInterface.h"
#include "Joystick/JoystickLib/JoystickThreaded.h"
GremseyGimbal::GremseyGimbal(GimbalInterface *parent) : GimbalInterface(parent)
{

}
void GremseyGimbal::setJoystick(JoystickThreaded* joystick){
    if(joystick == nullptr) return;
    m_joystick = joystick;
    connect(m_joystick,&JoystickThreaded::axisValueChanged,this,&GremseyGimbal::handleAxisValueChanged);
}
void GremseyGimbal::handleAxisValueChanged(int axisID, float value){
    if(!m_joystick->pic()){

        if(m_videoEngine == nullptr
                || m_context->m_lockMode == "FREE"){
            if(m_joystick->axisCount()<3) return;
            // send raw gimbal rate
            float maxRate = 1024/32768;
            float alphaSpeed = 3;
            float invertPan = 1;
            float invertTilt = 1;
            float panRate = m_joystick->axis(m_joystick->axisPan())->value();
            float tiltRate = m_joystick->axis(m_joystick->axisTilt())->value();
            float zoomRate = m_joystick->axis(m_joystick->axisZoom())->value();
            printf("%s [%f - %f - %f]\r\n",__func__, panRate,tiltRate,zoomRate);
            if(fabs(panRate) < 60) panRate = 0;
            if(fabs(tiltRate) < 60) tiltRate = 0;
            float x = invertPan * panRate*maxRate;
            float y = invertTilt * tiltRate*maxRate;
            float panRateScale = (alphaSpeed * m_context->m_hfov[m_context->m_sensorID] * x);
            float tiltRateScale = (alphaSpeed * m_context->m_hfov[m_context->m_sensorID] * y);
            setGimbalRate(panRateScale,tiltRateScale);
            setEOZoom("",zoomRate);
        }
    }
}
void GremseyGimbal::connectToGimbal(Config* config){
    if(config == nullptr)
        return;
    m_timerQueryZoom = new QTimer();
    m_timerQueryZoom->setInterval(100);
    m_timerQueryZoom->setSingleShot(false);
    m_config = config;
    m_gimbal = new GRGimbalController();
    printf("Gimbal[%s:%d]\r\n",m_config->value("Settings:GimbalIP:Value:data").toString().toStdString().c_str(),
            m_config->value("Settings:GimbalPortIn:Value:data").toInt());
    printf("Sensor[%s:%d]\r\n",m_config->value("Settings:SensorIP:Value:data").toString().toStdString().c_str(),
            m_config->value("Settings:SensorPortIn:Value:data").toInt());
    m_gimbal->setupTCP(m_config->value("Settings:GimbalIP:Value:data").toString(),
                        m_config->value("Settings:GimbalPortIn:Value:data").toInt());
    m_sensor = new TCPClient(m_config->value("Settings:SensorIP:Value:data").toString(),
            m_config->value("Settings:SensorPortIn:Value:data").toInt());
}
void GremseyGimbal::disconnectGimbal(){
    if(m_gimbal==nullptr)
        return;
}

void GremseyGimbal::changeSensor(QString sensorID){
    if(m_videoEngine!=nullptr){
        if(sensorID == "IR"){
            m_videoEngine->setVideo(m_config->value("Settings:StreamIR:Value:data").toString());
            m_videoEngine->start();
            m_context->setSensorID(1);
        }else{
            m_videoEngine->setVideo(m_config->value("Settings:StreamEO:Value:data").toString());
            m_videoEngine->start();
            m_context->setSensorID(0);
        }
    }
}
void GremseyGimbal::setEOZoom(QString command, int value){

}
void GremseyGimbal::setGimbalRate(float panRate,float tiltRate){
    printf("%s (panRate,tiltRate) = (%f,%f)\r\n",__func__,panRate,tiltRate);
    if(m_gimbal->getConnectionStatus() == 1){
        m_gimbal->set_Mode(e_control_gimbal_mode::GIMBAL_LOCK_MODE);
        m_gimbal->set_control(0,tiltRate,panRate);
    }
}
void GremseyGimbal::snapShot(){
    if(m_videoEngine!=nullptr){
        m_videoEngine->capture();
    }
}
void GremseyGimbal::setDigitalStab(bool enable){
    if(m_videoEngine!=nullptr){
        m_videoEngine->setStab(enable);
    }
}
void GremseyGimbal::setRecord(bool enable){

}
void GremseyGimbal::setShare(bool enable){

}
void GremseyGimbal::sendQueryZoom(){

}
void GremseyGimbal::handleQueryZoom(){

}
void GremseyGimbal::enableDigitalZoom(bool enable){

}
