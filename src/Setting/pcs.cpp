#include "pcs.h"

PCSConfig::PCSConfig(Config *parent) : Config(parent)
{
    //ctor
}

int PCSConfig::changeData(QString data,QString value){
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

void PCSConfig::print(){
    printf( "CAMERA: %s\r\n", mapData["CAM_CONTROL_IP"].toString().toStdString().c_str());
    printf( "    IN: %s\r\n", mapData["CAM_CONTROL_IN"].toString().toStdString().c_str());
    printf( " REPLY: %s\r\n", mapData["CAM_CONTROL_REP"].toString().toStdString().c_str());
    printf( "Stream: %s:%s\r\n",
            mapData["CAM_STREAM_IP"].toString().toStdString().c_str(),
            mapData["CAM_STREAM_PORT"].toString().toStdString().c_str());
    printf( "   FCS: %s\r\n", mapData["FCS_CONTROL_IP"].toString().toStdString().c_str());
    printf( "    IN: %s\r\n", mapData["FCS_CONTROL_IN"].toString().toStdString().c_str());
    printf( " REPLY: %s\r\n", mapData["FCS_CONTROL_REP"].toString().toStdString().c_str());
    printf( "TELEME: %s\r\n", mapData["TEL_CONTROL_IP"].toString().toStdString().c_str());
    printf( "  PORT: %s\r\n", mapData["TEL_CONTROL_PORT"].toString().toStdString().c_str());
    printf( "JOYSTK: %s\r\n", mapData["JOYSTICK_ID"].toString().toStdString().c_str());
    printf( "   PAN: %s\r\n", mapData["JOYSTICK_PAN_INVERT"].toString().toStdString().c_str());
    printf( "  TILT: %s\r\n", mapData["JOYSTICK_TILT_INVERT"].toString().toStdString().c_str());
    printf( "LANGUA: %s\r\n", mapData["LANGUAGE"].toString().toStdString().c_str());
}
int PCSConfig::readConfig(QString file){
    mapData.clear();
    if(m_doc.LoadFile(file.toStdString().c_str()) != tinyxml2::XML_SUCCESS){
        printf("Failed to load file\r\n");
        return -1;
    }
    XMLElement* camera = m_doc.FirstChildElement("CAMERA");

    mapData.insert("CAM_CONTROL_IP",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("CONTROL")->FirstChildElement("IP")->GetText())
                  );
    mapData.insert("CAM_CONTROL_IN",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("CONTROL")->FirstChildElement("IN")->GetText())
                  );
    mapData.insert("CAM_CONTROL_REP",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("CONTROL")->FirstChildElement("REPLY")->GetText())
                  );
    mapData.insert("SENSOR_CONTROL_IP",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("SENSOR")->FirstChildElement("IP")->GetText())
                  );
    mapData.insert("SENSOR_CONTROL_IN",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("SENSOR")->FirstChildElement("IN")->GetText())
                  );
    mapData.insert("CAM_STREAM_EO",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("STREAM_EO")->FirstChildElement("IP")->GetText())
                  );
    mapData.insert("CAM_STREAM_IR",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("STREAM_IR")->FirstChildElement("IP")->GetText())
                  );
    mapData.insert("CAM_CONNECT",
                   QString::fromLocal8Bit(
                       camera->FirstChildElement("CONNECT")->GetText())
                  );
    XMLElement *fcs = m_doc.FirstChildElement("FCS");
    mapData.insert("FCS_CONTROL_IP",
                   QString::fromLocal8Bit(
                       fcs->FirstChildElement("CONTROL")->FirstChildElement("IP")->GetText())
                  );
    mapData.insert("FCS_CONTROL_IN",
                   QString::fromLocal8Bit(
                       fcs->FirstChildElement("CONTROL")->FirstChildElement("IN")->GetText())
                  );
    mapData.insert("FCS_CONTROL_REP",
                   QString::fromLocal8Bit(
                       fcs->FirstChildElement("CONTROL")->FirstChildElement("REPLY")->GetText())
                  );
    XMLElement *tele = m_doc.FirstChildElement("TELEMETRY");
    mapData.insert("TEL_CONTROL_IP",
                   QString::fromLocal8Bit(
                       tele->FirstChildElement("IP")->GetText())
                  );
    mapData.insert("TEL_CONTROL_PORT",
                   QString::fromLocal8Bit(
                       tele->FirstChildElement("PORT")->GetText())
                  );
    XMLElement *joy = m_doc.FirstChildElement("JOYSTICK");
    mapData.insert("JOYSTICK_ID",
                   QString::fromLocal8Bit(
                       joy->FirstChildElement("PATH")->GetText())
                  );
    mapData.insert("JOYSTICK_PAN_INVERT",
                   QString::fromLocal8Bit(
                       joy->FirstChildElement("PAN")->GetText())
                  );
    mapData.insert("JOYSTICK_TILT_INVERT",
                   QString::fromLocal8Bit(
                       joy->FirstChildElement("TILT")->GetText())
                  );
    XMLElement *language = m_doc.FirstChildElement("LANGUAGE");
    mapData.insert("LANGUAGE",
                   QString::fromLocal8Bit(
                       language->FirstChildElement("ID")->GetText())
                  );
    XMLElement *map = m_doc.FirstChildElement("MAP");
    mapData.insert("MAP_FILE",
                   QString::fromLocal8Bit(
                       map->FirstChildElement("MAP_FILE")->GetText())
                  );
    mapData.insert("HEIGHT_FOLDER",
                   QString::fromLocal8Bit(
                       map->FirstChildElement("HEIGHT_FOLDER")->GetText())
                  );
    m_data = QVariant(mapData);
    print();
    return 0;
}
