#include "JoystickThreaded.h"

JoystickThreaded::JoystickThreaded(QObject* parent) :
    QObject(parent)
{
    m_task = new JoystickTask();
    m_workerThread = new QThread(NULL);
    m_task->moveToThread(m_workerThread);

    connect(m_task,SIGNAL(joystickConnected(bool)),this,SIGNAL(joystickConnected(bool)));
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
        // load axes
        tinyxml2::XMLElement * pListElementAxis = pElement->FirstChildElement("Axis");
        while(pListElementAxis!= nullptr){
            tinyxml2::XMLElement * pID = pListElementAxis->FirstChildElement("ID");
            tinyxml2::XMLElement * pFunc = pListElementAxis->FirstChildElement("Func");
            tinyxml2::XMLElement * pInvert = pListElementAxis->FirstChildElement("Inverted");
            mapAxis(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())),
                    QString::fromStdString(std::string(pInvert->GetText())) == "True"?true:false);
            mapAxisConfig(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())),
                    QString::fromStdString(std::string(pInvert->GetText())) == "True"?true:false);
            pListElementAxis = pListElementAxis->NextSiblingElement("Axis");
        }
        // load buttons
        tinyxml2::XMLElement * pListElementButton = pElement->FirstChildElement("Button");
        while(pListElementButton!= nullptr){
            tinyxml2::XMLElement * pID = pListElementButton->FirstChildElement("ID");
            tinyxml2::XMLElement * pFunc = pListElementButton->FirstChildElement("Func");
            mapButton(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())));
            mapButtonConfig(atoi(pID->GetText()),
                    QString::fromStdString(std::string(pFunc->GetText())));
            pListElementButton = pListElementButton->NextSiblingElement("Button");
        }
    }else{
        saveConfig();
    }
}
void JoystickThreaded::saveConfig(){
    if(m_mapFile.contains(".conf")){
        for(int i=0;i<m_axes.size(); i++){
            // copy
            m_axes.at(i)->setId(m_axesTemp.at(i)->id());
            m_axes.at(i)->setInverted(m_axesTemp.at(i)->inverted());
            m_axes.at(i)->setMapFunc(m_axesTemp.at(i)->mapFunc());

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
        for(int i=0;i<m_buttons.size(); i++){
            // copy
            m_buttons.at(i)->setId(m_buttonsTemp.at(i)->id());
            m_buttons.at(i)->setMapFunc(m_buttonsTemp.at(i)->mapFunc());
        }
        tinyxml2::XMLDocument xmlDoc;
        tinyxml2::XMLNode * pRoot = xmlDoc.NewElement("ArrayOfProperties");
        xmlDoc.InsertFirstChild(pRoot);
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
        }
        tinyxml2::XMLError eResult = xmlDoc.SaveFile(m_mapFile.toStdString().c_str());
    }
}
void JoystickThreaded::resetConfig(){
    if(m_axesTemp.size() > m_axisRoll){
        m_axesTemp.at(m_axisRoll)->setMapFunc("Roll");
        m_axesTemp.at(m_axisRoll)->setInverted(false);
    }
    if(m_axesTemp.size() > m_axisPitch){
        m_axesTemp.at(m_axisPitch)->setMapFunc("Pitch");
        m_axesTemp.at(m_axisPitch)->setInverted(false);
    }
    if(m_axesTemp.size() > m_axisYaw){
        m_axesTemp.at(m_axisYaw)->setMapFunc("Yaw");
        m_axesTemp.at(m_axisYaw)->setInverted(false);
    }
    if(m_axesTemp.size() > m_axisThrottle){
        m_axesTemp.at(m_axisThrottle)->setMapFunc("Throttle");
        m_axesTemp.at(m_axisThrottle)->setInverted(false);
    }

    for(int i=0; i< m_buttonsTemp.size(); i++){
        m_buttonsTemp.at(i)->setMapFunc("Unused");
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
            // update buttons
            m_buttons.clear();
            for(int i=0; i< m_task->m_joystick.m_buttons; i++)
                m_buttons.append(new JSButton(i));            
            // temp config
            // update axes
            m_axesTemp.clear();
            for(int i=0; i< m_task->m_joystick.m_axes; i++)
                m_axesTemp.append(new JSAxis(i));
            // update buttons
            m_buttonsTemp.clear();
            for(int i=0; i< m_task->m_joystick.m_buttons; i++)
                m_buttonsTemp.append(new JSButton(i));
            resetConfig();
            loadConfig();
            Q_EMIT axesConfigChanged();
            Q_EMIT buttonsConfigChanged();
        }
    }
}
void JoystickThreaded::mapAxis(int axisID, QString mapFunc, bool inverted){
    for(int i=0;i<m_axes.size(); i++){
        if(m_axes.at(i)->id() == axisID){
            m_axes.at(i)->setMapFunc(mapFunc);
            m_axes.at(i)->setInverted(inverted);
        }
    }
}
void JoystickThreaded::mapButton(int buttonID, QString mapFunc){
    for(int i=0;i<m_buttons.size(); i++){
        if(m_buttons.at(i)->id() == buttonID){
            m_buttons.at(i)->setMapFunc(mapFunc);
        }
    }
}
void JoystickThreaded::mapAxisConfig(int axisID, QString mapFunc, bool inverted){
    for(int i=0;i<m_axesTemp.size(); i++){
        if(m_axesTemp.at(i)->id() == axisID){
            m_axesTemp.at(i)->setMapFunc(mapFunc);
            m_axesTemp.at(i)->setInverted(inverted);
            break;
        }
    }
}
void JoystickThreaded::mapButtonConfig(int buttonID, QString mapFunc){
//    printf("%s %d %s\r\n",__func__,buttonID,mapFunc.toStdString().c_str());
    for(int i=0;i<m_buttonsTemp.size(); i++){
//        printf("Button[%d] = %s\r\n",i,
//               m_buttonsTemp.at(i)->mapFunc().toStdString().c_str());
        if(m_buttonsTemp.at(i)->id() == buttonID){
            m_buttonsTemp.at(i)->setMapFunc(mapFunc);
            break;
        }
    }
}
void JoystickThreaded::changeButtonState(int btnID,bool clicked){
    if(btnID < m_buttons.size()){
//        qDebug("Button %d is %s\n", btnID, !clicked ? "up" : "down");
        m_buttonsTemp[btnID]->setPressed(clicked);
        Q_EMIT buttonStateChanged(btnID,clicked);
    }
}
void JoystickThreaded::changeAxisValue(int axisID, float value){
    if(axisID < m_axes.size()){
//        qDebug("axisStateChanged %d value to %f\n", axisID, value);
        m_axesTemp[axisID]->setValue(value);
        m_axes[axisID]->setValue(value);
        Q_EMIT axisValueChanged(axisID,value);
    }
}
