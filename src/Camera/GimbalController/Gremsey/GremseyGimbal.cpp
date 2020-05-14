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
    setZoom(1.0f);
    setZoomMax(20);
    setZoomMin(1);
    setZoomTarget(1);
    Q_EMIT zoomCalculatedChanged(0,1);
}
void GremseyGimbal::setJoystick(JoystickThreaded* joystick){
    if(joystick == nullptr) return;
    m_joystick = joystick;
    connect(m_joystick,&JoystickThreaded::axisValueChanged,this,&GremseyGimbal::handleAxisValueChanged);
    connect(m_joystick,&JoystickThreaded::buttonStateChanged,this,&GremseyGimbal::handleButtonStateChanged);
    //    m_context->m_hfov[m_context->m_sensorID] = 62.9f;
}
void GremseyGimbal::handleButtonStateChanged(int buttonID, bool pressed){
    if(m_joystick != nullptr){
        if(buttonID >=0 && buttonID < m_joystick->buttonCount() && pressed){
            QString mapFunc = m_joystick->button(buttonID)->mapFunc();
            printf("button[%d] pressed=%s mapFunc=%s\r\n",
                   buttonID,pressed?"true":"false",mapFunc.toStdString().c_str());
            if(mapFunc == "EO/IR"){
                if(context()->m_sensorID == 0)
                    changeSensor("IR");
                else{
                    changeSensor("EO");
                }
            }else if(mapFunc == "SNAPSHOT"){
                snapShot();
            }else if(mapFunc == "VISUAL" || mapFunc == "FREE"){
                setLockMode(mapFunc);
            }else if(mapFunc.contains("PRESET")){
                setGimbalPreset(mapFunc);
            }else if(mapFunc == "DIGITAL_STAB"){
                setDigitalStab(!m_context->m_videoStabMode);
            }else if(mapFunc == "RECORD"){
                setRecord(!m_context->m_recording);
            }
        }
    }
}

void GremseyGimbal::handleAxisValueChanged(int axisID, float value){
    if(!m_joystick->pic()){
        if(m_joystick->axisCount()<3) return;
        // send raw gimbal rate
        float maxAxis = 32768.0f;
        float maxRate = 1024.0f/maxAxis;
        float deadZone = 160;
        float alphaSpeed = 2;
        float invertPan = m_joystick->invertPan();
        float invertTilt = m_joystick->invertTilt();
        float invertZoom = m_joystick->invertZoom();
        float panRate = m_joystick->axis(m_joystick->axisPan())->value();
        float tiltRate = m_joystick->axis(m_joystick->axisTilt())->value();
        float zoomRate = m_joystick->axis(m_joystick->axisZoom())->value();
        if(fabs(panRate) < deadZone) panRate = 0;
        if(fabs(tiltRate) < deadZone) tiltRate = 0;
        if(fabs(zoomRate) < deadZone) zoomRate = 0;
        float x = invertPan * panRate * maxRate;
        float y = invertTilt * tiltRate * maxRate;
        float z = invertZoom * zoomRate;
        float hfov = m_context->m_hfov[m_context->m_sensorID];
        float panRateScale = (alphaSpeed * hfov * x / m_context->m_hfovMax[m_context->m_sensorID]);
        float tiltRateScale = (alphaSpeed *  hfov * y / m_context->m_hfovMax[m_context->m_sensorID]);
//                printf("hfov[%.02f] hfovMax[%.02f] x[%.02f] y[%.02f] z[%.02f] panRateScale[%.02f] tiltRateScale[%.02f]\r\n",
//                       hfov,m_context->m_hfovMax[m_context->m_sensorID],
//                        x,y,z,
//                        panRateScale,tiltRateScale);

        if(m_videoEngine == nullptr
                || m_context->m_lockMode == "FREE"){
            setGimbalRate((panRateScale),(tiltRateScale));            
        }else{
            m_videoEngine->moveImage(-invertPan * panRate,
                                     -invertTilt * tiltRate,
                                     invertZoom * zoomRate);
        }
        if(m_joystick->axisZoom() == axisID){
            if(z > deadZone)
                setEOZoom("ZOOM_OUT",0);
            else if(z < -deadZone)
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
    setEOZoom("",1);
}
void GremseyGimbal::disconnectGimbal(){
    if(m_gimbal==nullptr)
        return;
}

void GremseyGimbal::changeSensor(QString sensorID){
    if(m_videoEngine!=nullptr){
        if(sensorID == "IR"){
            m_videoEngine->setVideo(m_config->value("Settings:StreamIR:Value:data").toString());
//            m_videoEngine->start();
            m_context->setSensorID(1);
        }else{
            m_videoEngine->setVideo(m_config->value("Settings:StreamEO:Value:data").toString());
//            m_videoEngine->start();
            m_context->setSensorID(0);
        }
    }
}
void GremseyGimbal::setEOZoom(QString command, float value){
    if(command == "ZOOM_IN"){
        printf("GremseyGimbal::%s %s\r\n",__func__,command.toStdString().c_str());
        m_sensor->sendRawData("068101040727FF");
    }else if(command == "ZOOM_OUT"){
        printf("GremseyGimbal::%s %s\r\n",__func__,command.toStdString().c_str());
        m_sensor->sendRawData("068101040737FF");
    }else if(command == "ZOOM_STOP"){
        printf("GremseyGimbal::%s %s\r\n",__func__,command.toStdString().c_str());
        m_sensor->sendRawData("068101040700FF");
    }else {
        printf("GremseyGimbal::%s %s\r\n",__func__,command.toStdString().c_str());
        if(value>=1 && value<= 240){
            int zoomPosition;
            int zoomBegin = value<=20? static_cast<int>(value)-1: static_cast<int>(value/20)-1+19;
            int zoomEnd = zoomBegin +1;
            if(value<=20)
                zoomPosition = m_zoom[zoomBegin] +
                        (value-zoomBegin-1)/(zoomEnd-zoomBegin)*(m_zoom[zoomEnd]-m_zoom[zoomBegin]);
            else{
                zoomPosition = m_zoom[zoomBegin] +
                        (value- ((zoomBegin-19)+1)*20)/((zoomEnd-zoomBegin) * 20.0f)*
                        (m_zoom[zoomEnd]-m_zoom[zoomBegin]);
            }
            char cmd[32];
            sprintf(cmd,"%04X",zoomPosition);

            QString zoomPositionRaw = QString::fromStdString(std::string(cmd));
            QString zoomPositionCommand = QString("0681010447") +
                    QString("0") + zoomPositionRaw[0]+
                    QString("0") + zoomPositionRaw[1]+
                    QString("0") + zoomPositionRaw[2]+
                    QString("0") + zoomPositionRaw[3]+"FF";
            //            printf("zoom[%.02f] equal %s from zoomBegin[%d][%04X] to zoomEnd[%d][%04X] zoomPositionCommand=%s\r\n",
            //                   value,cmd,zoomBegin,m_zoom[zoomBegin],zoomEnd,m_zoom[zoomEnd],
            //                   zoomPositionCommand.toStdString().c_str());
            m_sensor->sendRawData(zoomPositionCommand);
        }
    }
}
void GremseyGimbal::setGimbalRate(float panRate,float tiltRate){
    if(m_gimbal->getConnectionStatus() == 1){
        m_gimbal->set_Mode(e_control_gimbal_mode::GIMBAL_LOCK_MODE);
        m_gimbal->set_control(0,
                              static_cast<int>(tiltRate),
                              static_cast<int>(panRate));
    }
}
void GremseyGimbal::snapShot(){
    if(m_videoEngine!=nullptr){
        m_videoEngine->capture();
        Q_EMIT functionHandled("Snapshot has been taken!");
    }
}
void GremseyGimbal::changeTrackSize(float trackSize){
    if(m_videoEngine!=nullptr){
        m_videoEngine->changeTrackSize((int)trackSize);
    }
}
void GremseyGimbal::setDigitalStab(bool enable){
    if(m_videoEngine!=nullptr){
        m_videoEngine->setStab(enable);
        m_context->m_videoStabMode = enable;
    }
}
void GremseyGimbal::setLockMode(QString mode, QPoint location){
    m_context->m_lockMode = mode;
    if(mode == "FREE"){
    }else if(mode == "TRACK"){
        m_videoEngine->setTrackAt(0,
                                  m_videoEngine->sourceSize().width()/2,
                                  m_videoEngine->sourceSize().height()/2,
                                  m_videoEngine->sourceSize().width(),
                                  m_videoEngine->sourceSize().height());
        resetTrackParam();
    }else if(mode == "VISUAL"){

    }
}
void GremseyGimbal::setRecord(bool enable){
    m_context->m_recording = !m_context->m_recording;
    if(m_videoEngine != nullptr){
        m_videoEngine->setRecord(m_context->m_recording);
    }
}
void GremseyGimbal::setShare(bool enable){
    m_context->m_gcsShare = !m_context->m_gcsShare;
    if(m_videoEngine != nullptr){
        m_videoEngine->setShare(m_context->m_gcsShare);
    }
}
void GremseyGimbal::sendQueryZoom(){
    m_sensor->sendRawData("0681090447FF");
}
void GremseyGimbal::handleQuery(QString data){
    if(data.startsWith("9050") && data.size() == 14 && data.endsWith("FF"))
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
    }
}
void GremseyGimbal::enableDigitalZoom(bool enable){

}
