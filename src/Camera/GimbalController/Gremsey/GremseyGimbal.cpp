#include "GremseyGimbal.h"
#include "GimbalControl.h"
#include "SensorController.h"
#include "Camera/VideoEngine/VideoEngineInterface.h"
#include "Joystick/JoystickLib/JoystickThreaded.h"
GremseyGimbal::GremseyGimbal(GimbalInterface *parent) : GimbalInterface(parent)
{

    m_zoom.append(0x0000); //1x
    m_zoom.append(0x0DC1); //2x
    m_zoom.append(0x186C); //3x
    m_zoom.append(0x2015); //4x
    m_zoom.append(0x2594); //5x
    m_zoom.append(0x29B7); //6x
    m_zoom.append(0x2CFB); //7x
    m_zoom.append(0x2FB0); //8x
    m_zoom.append(0x320C); //9x
    m_zoom.append(0x342D); //10x
    m_zoom.append(0x3608); //11x
    m_zoom.append(0x37AA); //12x
    m_zoom.append(0x391C); //13x
    m_zoom.append(0x3A66); //14x
    m_zoom.append(0x3B90); //15x
    m_zoom.append(0x3C9C); //16x
    m_zoom.append(0x3D91); //17x
    m_zoom.append(0x3E72); //18x
    m_zoom.append(0x3F40); //19x
    m_zoom.append(0x4000); // 20x * 1
    m_zoom.append(0x6000); // 20x * 2
    m_zoom.append(0x6AAB); // 20x * 3
    m_zoom.append(0x7000); // 20x * 4
    m_zoom.append(0x7334); // 20x * 5
    m_zoom.append(0x7556); // 20x * 6
    m_zoom.append(0x76DC); // 20x * 7
    m_zoom.append(0x7800); // 20x * 8
    m_zoom.append(0x78E4); // 20x * 9
    m_zoom.append(0x799A); // 20x * 10
    m_zoom.append(0x7A2F); // 20x * 11
    m_zoom.append(0x7AC0); // 20x * 12
    for(int i=0; i< m_zoom.size(); i++){
        if(i<20){
            m_mapZoom[m_zoom[i]] = i+1;
        }else{
            m_mapZoom[m_zoom[i]] = 20*(i-20+2);
        }
    }
    m_context->m_hfovMax[0] = 70.2f;
    m_context->m_hfovMax[1] = 17.7f;
    m_context->m_hfovMin[0] = 4.2f;
    m_context->m_hfovMin[1] = 17.7f;
    m_context->m_hfov[0] = m_context->m_hfovMax[0];
    m_context->m_hfov[1] = 17.7f;
    m_context->m_zoom[0] = 1;
}
void GremseyGimbal::setJoystick(JoystickThreaded* joystick){
    if(joystick == nullptr) return;
    m_joystick = joystick;
    connect(m_joystick,&JoystickThreaded::axisValueChanged,this,&GremseyGimbal::handleAxisValueChanged);
//    m_context->m_hfov[m_context->m_sensorID] = 62.9f;
}
void GremseyGimbal::handleAxisValueChanged(int axisID, float value){
    if(!m_joystick->pic()){
        if(m_videoEngine == nullptr
                || m_context->m_lockMode == "FREE"){
            if(m_joystick->axisCount()<3) return;
            // send raw gimbal rate
            float maxAxis = 32768.0f;
            float maxRate = 1024.0f/maxAxis;
            float deadZone = 60;
            float alphaSpeed = 1;
            float invertPan = -1;
            float invertTilt = 1;
            float panRate = m_joystick->axis(m_joystick->axisPan())->value();
            float tiltRate = m_joystick->axis(m_joystick->axisTilt())->value();
            float zoomRate = m_joystick->axis(m_joystick->axisZoom())->value();
            printf("%s maxRate = %f before [%f - %f - %f]\r\n",__func__, maxRate,panRate,tiltRate,zoomRate);
            if(fabs(panRate) < 60) panRate = 0;
            if(fabs(tiltRate) < 60) tiltRate = 0;
            printf("%s after [%f - %f - %f]\r\n",__func__, panRate,tiltRate,zoomRate);
            float x = invertPan * panRate * maxRate;
            float y = invertTilt * tiltRate * maxRate;
            float hfov = m_context->m_hfov[m_context->m_sensorID];
            if(hfov<1.0){
                 hfov = 62.9f;
            }
            float panRateScale = (alphaSpeed * hfov / 62.9* x);
            float tiltRateScale = (alphaSpeed *  hfov / 62.9* y);
            setGimbalRate((panRateScale),(tiltRateScale));
            if(zoomRate < -deadZone)
                setEOZoom("ZOOM_OUT",0);
            else if(zoomRate >deadZone)
                setEOZoom("ZOOM_IN",0);
            else
                setEOZoom("ZOOM_STOP",0);
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
    m_sensor = new SensorController();
    m_sensor->connectToHost(m_config->value("Settings:SensorIP:Value:data").toString(),
            m_config->value("Settings:SensorPortIn:Value:data").toInt());
    connect(m_timerQueryZoom,&QTimer::timeout,this,&GremseyGimbal::sendQueryZoom);
    connect(m_sensor,&SensorController::dataReceived,this,&GremseyGimbal::handleQuery);
    m_timerQueryZoom->start();
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
    if(command == "ZOOM_IN"){
        m_sensor->sendRawData("068101040727FF");
    }else if(command == "ZOOM_OUT"){
        m_sensor->sendRawData("068101040737FF");
    }else if(command == "ZOOM_STOP"){
        m_sensor->sendRawData("068101040700FF");
    }else {

    }
}
void GremseyGimbal::setGimbalRate(float panRate,float tiltRate){
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
    m_sensor->sendRawData("0681090447FF");
}
void GremseyGimbal::handleQuery(QString data){
    if(data.contains("9050"))
    {
        // zoom position
        QString zoomPosition = QString(data[5])+QString(data[7])+QString(data[9])+QString(data[11]);
        int zoomPos = zoomPosition.toInt(nullptr,16);
        float zoomPosF = static_cast<float>(zoomPos);
        for(int i=0; i< m_zoom.size()-1; i++){
            if(zoomPos >= m_zoom[i] && zoomPos <= m_zoom[i+1]){
                zoomPosF = static_cast<float>(m_mapZoom[m_zoom[i]]) +
                        static_cast<float>(zoomPos - m_zoom[i])/static_cast<float>(m_zoom[i+1]-m_zoom[i]) *
                        (static_cast<float>(m_mapZoom[m_zoom[i+1]]) - static_cast<float>(m_mapZoom[m_zoom[i]]));

                m_context->m_zoom[0] = zoomPosF;
                m_context->m_hfov[0] = atanf(tan(m_context->m_hfovMax[0]/2/180*M_PI)/zoomPosF)/M_PI*180*2;
                break;
            }
        }
//        printf("zoomPosition = %s %04X X%.02f %.02f deg\r\n",zoomPosition.toStdString().c_str(),zoomPos,zoomPosF,m_context->m_hfov[0]);
    }
}
void GremseyGimbal::enableDigitalZoom(bool enable){

}
