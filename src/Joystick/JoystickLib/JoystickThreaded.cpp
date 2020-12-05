#include "JoystickThreaded.h"

JoystickThreaded::JoystickThreaded(QObject* parent) :
    QObject(parent)
{
    m_task = new JoystickTask();
    m_workerThread = new QThread(NULL);
    m_task->moveToThread(m_workerThread);

    connect(m_task,SIGNAL(joystickConnected(bool)),this,SLOT(setConnected(bool)));
    connect(m_task,SIGNAL(joystickConnected(bool)),this,SLOT(updateButtonAxis(bool)));
    connect(m_task,SIGNAL(btnClicked(int,bool)),this,SLOT(changeButtonState(int,bool)));
    connect(m_task,SIGNAL(axisStateChanged(int,float)),this,SLOT(changeAxisValue(int,float)));
    //
//    connect(m_task,SIGNAL(btnClicked(int,bool)),this,SIGNAL(axesChanged()));
//    connect(m_task,SIGNAL(axisStateChanged(int,float)),this,SIGNAL(buttonsChanged()));
    //
    connect(m_workerThread, SIGNAL(started()), m_task, SLOT(doWork()));
    qDebug("joystick connect done");
}
/**/
JoystickThreaded::~JoystickThreaded()
{
    m_task->m_stop = true;
    m_workerThread->quit();
    if(!m_workerThread->wait(1000)){
         m_workerThread->terminate();
        m_workerThread->wait(1000);
    }
    m_workerThread->deleteLater();
    m_task->deleteLater();
    printf("JoystickThreaded destroyed\r\n");
}

void JoystickThreaded::start()
{
    if(m_workerThread->isRunning() == false)
        m_workerThread->start();
    else{
        pause(false);
    }
}

void JoystickThreaded::pause(bool _pause)
{
    m_task->pause(_pause);
}
void JoystickThreaded::stop()
{
    m_task->m_stop = true;
    m_workerThread->quit();

    if(!m_workerThread->wait(1000)){
         m_workerThread->terminate();
        m_workerThread->wait(1000);
    }
}
void JoystickThreaded::setJoyID(QString joyID){
    m_task->setJoyID(joyID);
}
void JoystickThreaded::loadConfig(){
    tinyxml2::XMLDocument m_doc;
    tinyxml2::XMLError res = m_doc.LoadFile(m_mapFile.toStdString().c_str());
    if(res == tinyxml2::XML_SUCCESS){
        tinyxml2::XMLElement * pElement = m_doc.FirstChildElement("ArrayOfProperties");
        // load use joystick
        tinyxml2::XMLElement * pElementUseJoystick = pElement->FirstChildElement("UseJoystick");
        if(pElementUseJoystick!= nullptr){
            m_useJoystick = QString::fromStdString(std::string(pElementUseJoystick->GetText())) == "True"?true:false;
            Q_EMIT useJoystickChanged(m_useJoystick);
        }
        // load axes flight
        tinyxml2::XMLElement * pListElementAxis = pElement->FirstChildElement("Axis");
        while(pListElementAxis!= nullptr){
            tinyxml2::XMLElement * pID = pListElementAxis->FirstChildElement("ID");
            tinyxml2::XMLElement * pFunc = pListElementAxis->FirstChildElement("Func");
            tinyxml2::XMLElement * pInvert = pListElementAxis->FirstChildElement("Inverted");
            mapAxis(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())),
                    QString::fromStdString(std::string(pInvert->GetText())) == "True"?true:false,
                    true);
            pListElementAxis = pListElementAxis->NextSiblingElement("Axis");
        }
        // load axes Cam
//        printf("Load Axes cam\r\n");
        tinyxml2::XMLElement * pListElementAxisCam = pElement->FirstChildElement("AxisCam");
        while(pListElementAxisCam!= nullptr){
            tinyxml2::XMLElement * pID = pListElementAxisCam->FirstChildElement("ID");
            tinyxml2::XMLElement * pFunc = pListElementAxisCam->FirstChildElement("Func");
            tinyxml2::XMLElement * pInvert = pListElementAxisCam->FirstChildElement("Inverted");
            mapAxis(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())),
                    QString::fromStdString(std::string(pInvert->GetText())) == "True"?true:false,
                    true,true);
            pListElementAxisCam = pListElementAxisCam->NextSiblingElement("AxisCam");
        }
        // load buttons
        tinyxml2::XMLElement * pListElementButton = pElement->FirstChildElement("Button");
        while(pListElementButton!= nullptr){
            tinyxml2::XMLElement * pID = pListElementButton->FirstChildElement("ID");
            tinyxml2::XMLElement * pFunc = pListElementButton->FirstChildElement("Func");
            mapButton(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())));
            pListElementButton = pListElementButton->NextSiblingElement("Button");
        }
        for(int i=0;i<m_axes.size(); i++){
            if(m_axes.at(i)->mapFunc() == "Roll"){
                m_axisRoll = m_axes.at(i)->id();
            }else if(m_axes.at(i)->mapFunc() == "Pitch"){
                m_axisPitch = m_axes.at(i)->id();
            }else if(m_axes.at(i)->mapFunc() == "Yaw"){
                m_axisYaw = m_axes.at(i)->id();
            }else if(m_axes.at(i)->mapFunc() == "Throttle"){
                m_axisThrottle = m_axes.at(i)->id();
            }
        }
        for(int i=0;i<m_axesCam.size(); i++){
            if(m_axesCam.at(i)->mapFunc() == "Pan"){
                m_axisPan = m_axes.at(i)->id();
            }else if(m_axesCam.at(i)->mapFunc() == "Tilt"){
                m_axisTilt = m_axes.at(i)->id();
            }else if(m_axesCam.at(i)->mapFunc() == "Zoom"){
                m_axisZoom = m_axes.at(i)->id();
            }
        }
        for(int i=0;i< m_buttons.size(); i++){
            if(m_buttons.at(i)->mapFunc()=="PIC/CIC"
                    || m_buttons.at(i)->mapFunc()=="CIC/PIC"){
                m_butonPICCIC = m_buttons.at(i)->id();
            }
        }
    }else{
        saveConfig();
    }
}
void JoystickThreaded::saveConfig(){
    if(m_mapFile.contains(".conf")){
        for(int i=0;i<m_axes.size(); i++){
            // copy
            m_axes.at(i)->saveConfig();
            if(m_axes.at(i)->mapFunc() == "Roll"){
                m_axisRoll = m_axes.at(i)->id();
            }else if(m_axes.at(i)->mapFunc() == "Pitch"){
                m_axisPitch = m_axes.at(i)->id();
            }else if(m_axes.at(i)->mapFunc() == "Yaw"){
                m_axisYaw = m_axes.at(i)->id();
            }else if(m_axes.at(i)->mapFunc() == "Throttle"){
                m_axisThrottle = m_axes.at(i)->id();
            }
        }
        for(int i=0;i<m_axesCam.size(); i++){
            // copy
            m_axesCam.at(i)->saveConfig();
            if(m_axesCam.at(i)->mapFunc() == "Pan"){
                m_axisPan = m_axesCam.at(i)->id();
            }else if(m_axesCam.at(i)->mapFunc() == "Tilt"){
                m_axisTilt = m_axesCam.at(i)->id();
            }else if(m_axesCam.at(i)->mapFunc() == "Zoom"){
                m_axisZoom = m_axesCam.at(i)->id();
            }
        }
        for(int i=0;i<m_buttons.size(); i++){
            // copy
            m_buttons.at(i)->saveConfig();
            if(m_buttons.at(i)->mapFunc() == "PIC/CIC"
                     || m_buttons.at(i)->mapFunc()=="CIC/PIC"){
                m_butonPICCIC = i;
            }
        }
        tinyxml2::XMLDocument xmlDoc;
        tinyxml2::XMLNode * pRoot = xmlDoc.NewElement("ArrayOfProperties");
        xmlDoc.InsertFirstChild(pRoot);
        tinyxml2::XMLElement * pElementUseJoystick = xmlDoc.NewElement("UseJoystick");
        pElementUseJoystick->SetText(m_useJoystick?"True":"False");
        pRoot->InsertEndChild(pElementUseJoystick);
        Q_EMIT useJoystickChanged(m_useJoystick);
        for(int i=0; i< m_axes.size(); i++){
            JSAxis *tmp = m_axes.at(i);
            tinyxml2::XMLElement * pElement = xmlDoc.NewElement("Axis");
            tinyxml2::XMLElement * pID = xmlDoc.NewElement("ID");
            pID->SetText(std::to_string(tmp->id()).c_str());
            pElement->InsertEndChild(pID);
            tinyxml2::XMLElement * pFunc = xmlDoc.NewElement("Func");
            pFunc->SetText(tmp->mapFunc().toStdString().c_str());
            pElement->InsertEndChild(pFunc);
            tinyxml2::XMLElement * pInvert = xmlDoc.NewElement("Inverted");
            pInvert->SetText(tmp->inverted()?"True":"False");
            pElement->InsertEndChild(pInvert);
            pRoot->InsertEndChild(pElement);
            printf("Save axis[%d] [%s] [%s]\r\n",i,tmp->mapFunc().toStdString().c_str(),tmp->inverted()?"True":"False");
        }
        for(int i=0; i< m_axesCam.size(); i++){
            JSAxis *tmp = m_axesCam.at(i);
            tinyxml2::XMLElement * pElement = xmlDoc.NewElement("AxisCam");
            tinyxml2::XMLElement * pID = xmlDoc.NewElement("ID");
            pID->SetText(std::to_string(tmp->id()).c_str());
            pElement->InsertEndChild(pID);
            tinyxml2::XMLElement * pFunc = xmlDoc.NewElement("Func");
            pFunc->SetText(tmp->mapFunc().toStdString().c_str());
            pElement->InsertEndChild(pFunc);
            tinyxml2::XMLElement * pInvert = xmlDoc.NewElement("Inverted");
            pInvert->SetText(tmp->inverted()?"True":"False");
            pElement->InsertEndChild(pInvert);
            pRoot->InsertEndChild(pElement);
            printf("Save axisCam[%d] [%s] [%s]\r\n",i,tmp->mapFunc().toStdString().c_str(),tmp->inverted()?"True":"False");
        }
        for(int i=0; i< m_buttons.size(); i++){
            JSButton *tmp = m_buttons.at(i);
            tinyxml2::XMLElement * pElement = xmlDoc.NewElement("Button");
            tinyxml2::XMLElement * pID = xmlDoc.NewElement("ID");
            pID->SetText(std::to_string(tmp->id()).c_str());
            pElement->InsertEndChild(pID);
            tinyxml2::XMLElement * pFunc = xmlDoc.NewElement("Func");
            pFunc->SetText(tmp->mapFunc().toStdString().c_str());
            pElement->InsertEndChild(pFunc);
            pRoot->InsertEndChild(pElement);
            printf("Save button[%d] [%s] \r\n",i,tmp->mapFunc().toStdString().c_str());
        }
        tinyxml2::XMLError eResult = xmlDoc.SaveFile(m_mapFile.toStdString().c_str());
    }
}
void JoystickThreaded::resetConfig(){
    // flight
    if(m_axes.size() > m_axisRoll){
        m_axes.at(m_axisRoll)->setMapFunc("Roll");
        m_axes.at(m_axisRoll)->setInverted(false);
    }
    if(m_axes.size() > m_axisPitch){
        m_axes.at(m_axisPitch)->setInverted(false);
        m_axes.at(m_axisPitch)->setMapFunc("Pitch");
    }
    if(m_axes.size() > m_axisYaw){
        m_axes.at(m_axisYaw)->setMapFunc("Yaw");
        m_axes.at(m_axisYaw)->setInverted(false);
    }
    if(m_axes.size() > m_axisThrottle){
        m_axes.at(m_axisThrottle)->setMapFunc("Throttle");
        m_axes.at(m_axisThrottle)->setInverted(false);
    }
    // camera
    if(m_axes.size() > m_axisPan){
        m_axes.at(m_axisPan)->setInverted(false);
        m_axes.at(m_axisPan)->setMapFunc("Pan");
    }
    if(m_axes.size() > m_axisTilt){
        m_axes.at(m_axisTilt)->setInverted(false);
        m_axes.at(m_axisTilt)->setMapFunc("Tilt");
    }
    if(m_axes.size() > m_axisZoom){
        m_axes.at(m_axisZoom)->setInverted(false);
        m_axes.at(m_axisZoom)->setMapFunc("Zoom");
    }
    // buttons
    for(int i=0; i< m_buttons.size(); i++){
        m_buttons.at(i)->setMapFunc("Unused");
    }
}
void JoystickThreaded::updateButtonAxis(bool connected){
    if(connected){
        if(m_task->m_joystick.m_axes != m_axes.size() ||
                m_task->m_joystick.m_buttons != m_buttons.size()){
//            printf("updateButtonAxis ============================== axis[%d] button[%d]\r\n",
//                   m_task->m_joystick.m_axes,
//                   m_task->m_joystick.m_buttons);
            // main config
            // update axes
            m_axes.clear();
            for(int i=0; i< m_task->m_joystick.m_axes; i++)
                m_axes.append(new JSAxis(i));            
            m_axesCam.clear();
            for(int i=0; i< m_task->m_joystick.m_axes; i++)
                m_axesCam.append(new JSAxis(i));
            // update buttons
            m_buttons.clear();
            for(int i=0; i< m_task->m_joystick.m_buttons; i++)
                m_buttons.append(new JSButton(i));
            resetConfig();
            loadConfig();
            Q_EMIT axesChanged();
            Q_EMIT axesCamChanged();
            Q_EMIT buttonsChanged();
            Q_EMIT buttonAxisLoaded();
        }
    }
}
void JoystickThreaded::mapAxis(int axisID, QString mapFunc, bool inverted, bool saveCurrent, bool axisCam){
    if(!axisCam){
        for(int i=0;i<m_axes.size(); i++){
            if(m_axes.at(i)->id() == axisID){
                m_axes.at(i)->setMapFuncConfig(mapFunc);
                m_axes.at(i)->setInvertedConfig(inverted);
                if(saveCurrent){
                    m_axes.at(i)->saveConfig();
                }
            }else{
                if(mapFunc == m_axes.at(i)->mapFuncConfig()){
                    m_axes.at(i)->setMapFuncConfig("Unused");
                    if(saveCurrent){
                        m_axes.at(i)->saveConfig();
                    }
                }
            }
        }
    }else{
        for(int i=0;i<m_axesCam.size(); i++){
            if(m_axesCam.at(i)->id() == axisID){
                m_axesCam.at(i)->setMapFuncConfig(mapFunc);
                m_axesCam.at(i)->setInvertedConfig(inverted);
                if(saveCurrent){
                    m_axesCam.at(i)->saveConfig();
                }
            }else{
                if(mapFunc == m_axesCam.at(i)->mapFuncConfig()){
                    m_axesCam.at(i)->setMapFuncConfig("Unused");
                    if(saveCurrent){
                        m_axesCam.at(i)->saveConfig();
                    }
                }
            }
        }
    }


}
void JoystickThreaded::mapButton(int buttonID, QString mapFunc, bool saveCurrent){
    for(int i=0;i<m_buttons.size(); i++){
        if(m_buttons.at(i)->id() == buttonID){
            m_buttons.at(i)->setMapFuncConfig(mapFunc);
            if(saveCurrent)
                m_buttons.at(i)->saveConfig();
        }else{
            if(mapFunc.contains("PIC")&&
                    m_buttons.at(i)->mapFuncConfig().contains("PIC")){
                m_buttons.at(i)->setMapFuncConfig("Unused");
                if(saveCurrent)
                    m_buttons.at(i)->saveConfig();
            }
        }
    }
}

void JoystickThreaded::setInvertCam(QString camFunc,bool invert){
    if(camFunc == "PAN"){
        mapAxis(m_axisPan,camFunc,invert,true,true);
    }else if(camFunc == "TILT"){
        mapAxis(m_axisTilt,camFunc,invert,true,true);
    }else if(camFunc == "ZOOM"){
        mapAxis(m_axisZoom,camFunc,invert,true,true);
    }
}
void JoystickThreaded::changeButtonState(int btnID,bool clicked){
    if(btnID < m_buttons.size()){
//        qDebug("Button %d is %s\n", btnID, !clicked ? "up" : "down");
        if(btnID == m_butonPICCIC){
            m_buttons[btnID]->setPressed(m_buttons[btnID]->mapFunc() == "PIC/CIC"?clicked:!clicked);
            setPIC(m_buttons[btnID]->mapFunc() == "PIC/CIC"?clicked:!clicked);
            Q_EMIT buttonStateChanged(btnID,m_buttons[btnID]->mapFunc() == "PIC/CIC"?clicked:!clicked);
        }else{
            m_buttons[btnID]->setPressed(clicked);
            Q_EMIT buttonStateChanged(btnID,clicked);
        }
    }
}
void JoystickThreaded::changeAxisValue(int axisID, float value){
    if(axisID < m_axes.size()){
        m_axes[axisID]->setValue(value);
        m_axesCam[axisID]->setValue(value);
        Q_EMIT axisValueChanged(axisID,value);

    }
}
