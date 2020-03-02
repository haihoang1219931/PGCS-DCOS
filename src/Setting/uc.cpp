#include "uc.h"

UCConfig::UCConfig(Config *parent) : Config(parent)
{
    //ctor
}

int UCConfig::changeData(QString data,QString value){
    if(data == "CAM_CONNECT")
        m_doc.FirstChildElement( "CAMERA" )->
                FirstChildElement( "CONNECT" )->
                SetText((const char*)value.toStdString().c_str());
    else if(data == "CAM_CONTROL_IP")
        m_doc.FirstChildElement( "CAMERA" )->
                FirstChildElement( "CONTROL" )->
                FirstChildElement("IP")->
                SetText((const char*)value.toStdString().c_str());
    else if(data == "JOYSTICK_ID")
        m_doc.FirstChildElement( "JOYSTICK" )->
                FirstChildElement( "PATH" )->
                SetText((const char*)value.toStdString().c_str());
    return 0;
}

void UCConfig::print(){

}
