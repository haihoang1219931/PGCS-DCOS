#include "config.h"

Config::Config(QObject *parent) : QObject(parent)
{
    //ctor
}

int Config::changeData(QString data,QString value){
    printf("%s[%s]=%s\r\n",
           __func__,
           data.toStdString().c_str(),
           value.toStdString().c_str());
    if(m_settings != nullptr){
        QVariantMap settingsMap;
        for(tinyxml2::XMLElement* child = m_settings->FirstChildElement();
            child != nullptr; child = child->NextSiblingElement()){
            if(QString::fromUtf8(child->Attribute("Name")) == data){
                if(child->FirstChildElement("Value") != nullptr){
                    child->FirstChildElement("Value")->
                        SetText((const char*)value.toStdString().c_str());
                    saveConfig(m_fileConfig);
                    break;
                }
            }
        }
    }
    return 0;
}

void Config::print(){
}
int Config::readConfig(QString file){    
    m_fileConfig = file;
    m_mapData.clear();
    m_paramsModel.clear();
    if(m_doc.LoadFile(file.toStdString().c_str()) != tinyxml2::XML_SUCCESS){
        createFile(file);
        m_doc.LoadFile(file.toStdString().c_str());
    }
    m_settings = m_doc.FirstChildElement("SettingsFile")->FirstChildElement("Settings");
    if(m_settings != nullptr){
        QVariantMap settingsMap;
        for(tinyxml2::XMLElement* child = m_settings->FirstChildElement();
            child != nullptr; child = child->NextSiblingElement()){
            QVariantMap settingMap;
            QVariantMap attributesMap;
            if(child->Attribute("Name")!=nullptr){
                attributesMap.insert("Name",QString::fromUtf8(child->Attribute("Name")));
                attributesMap.insert("Type",QString::fromUtf8(child->Attribute("Type")));
                attributesMap.insert("Scope",QString::fromUtf8(child->Attribute("Scope")));
                QVariantMap valueMap;
                if(child->FirstChildElement("Value")!= nullptr){
                    valueMap.insert("data",QString::fromUtf8(child->FirstChildElement("Value")->GetText()));
                    valueMap.insert("profile",QString::fromUtf8(child->Attribute("Profile")));
                    attributesMap.insert("Name",QString::fromUtf8(child->Attribute("Name")));
                    settingMap.insert("Attributes",attributesMap);
                    settingMap.insert("Value",valueMap);
                    settingsMap.insert(QString::fromUtf8(child->Attribute("Name")),settingMap);
                    ConfigElement *configElement = new ConfigElement(attributesMap["Name"].toString(),
                                        valueMap["data"].toString());
                    m_paramsModel.append(configElement);
                }
            }
        }
        m_mapData.insert("Settings",settingsMap);
    }else{
        printf("Failed to find element\r\n");
    }
    m_data = QVariant(m_mapData);
    Q_EMIT paramsModelChanged();
    return 0;
}
QVariant Config::getData(){
    return m_data;
}
int Config::saveConfig(QString file){
    if(file != "")
        m_doc.SaveFile(file.toStdString().c_str(),false);
    return 0;
}
QVariant Config::value(QString pattern){
    QVariant value;
    QStringList listKey = pattern.split(":");
    QVariantMap valueMap = m_mapData;
    int i=0;
    while(i<listKey.size()-1){
        valueMap = qvariant_cast<QVariantMap>(valueMap.value(listKey[i]));
        i++;
        if(i == listKey.size()-1){
            value = valueMap.value(listKey[i]);
            break;
        }
    }
    return value;
}
void Config::createFile(QString fileName){
    if(fileName.contains("conf/")){
        system("/bin/mkdir -p conf");
    }
    printf("%s [%s]\r\n",__func__,fileName.toStdString().c_str());
    if(fileName.toLower().contains("fcs.conf")){
        std::string text =
                    "<?xml version='1.0' encoding='utf-8'?>\r\n"
                    "<SettingsFile>\r\n"
                        "\t<Settings>\r\n"
                            "\t\t<Setting Name=\"LinkName\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">MP</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"LinkType\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">RAGAS</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"Baudrate\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">57600</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"PortName\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">/dev/ttyACM0</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"PilotIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">192.168.0.220</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"PilotPortIn\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">20002</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"PilotPortOut\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">40001</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"PingInterval\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">2000</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"MulticastIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">232.4.130.147</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"MulticastPort\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">40002</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"MapDefault\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">HoaLac.tpk</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"ElevationDataPath\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">ElevationData-H1</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleLocalIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">192.168.0.30</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleLocalPort\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">23</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleLocalUser\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">admin</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleLocalPass\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">ttuav</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleRemoteIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">192.168.0.220</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleRemotePort\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">23</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleRemoteUser\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">admin</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"TeleRemotePass\" Type=\"System.String\" Scope=\"User\">\r\n"
                                "\t\t\t<Value Profile=\"(Default)\">ttuav</Value>\r\n"
                            "\t\t</Setting>\r\n"
                        "\t</Settings>\r\n"
                    "</SettingsFile>\r\n";
        FileController::addLine(fileName.toStdString(),text);
    }else if(fileName.toLower().contains("trk.conf")){
        std::string text =
                    "<?xml version='1.0' encoding='utf-8'?>\r\n"
                    "<SettingsFile>\r\n"
                      "\t<Settings>\r\n"
                        "\t\t<Setting Name=\"LinkName\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">MP Comm</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"LinkType\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">UDP</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"Baudrate\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">57600</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"PortName\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">/dev/ttyUSB0</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"PilotIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">192.168.0.2</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"PilotPortIn\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">20012</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"PilotPortOut\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">40001</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"PingInterval\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">2000</Value>\r\n"
                        "\t\t</Setting>\r\n"
                      "\t</Settings>\r\n"
                    "</SettingsFile>\r\n";
        FileController::addLine(fileName.toStdString(),text);
    }else if(fileName.toLower().contains("uc.conf")){
        printf("Create UC config\r\n");
        std::string text =
                    "<?xml version='1.0' encoding='utf-8'?>\r\n"
                    "<SettingsFile>\r\n"
                        "\t<Settings>\r\n"
                            "\t\t<Setting Name=\"UCServerAddress\" Type=\"System.String\" Scope=\"User\">\r\n"
                              "\t\t\t<Value Profile=\"(Default)\">192.168.43.173</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"UCServerPort\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                              "\t\t\t<Value Profile=\"(Default)\">3000</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"UCServerName\" Type=\"System.String\" Scope=\"User\">\r\n"
                              "\t\t\t<Value Profile=\"(Default)\">FCS 01</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"UCRoomName\" Type=\"System.String\" Scope=\"User\">\r\n"
                              "\t\t\t<Value Profile=\"(Default)\">Quard 01</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"UCStreamSource\" Type=\"System.String\" Scope=\"User\">\r\n"
                              "\t\t\t<Value Profile=\"(Default)\">rtsp://127.0.0.1:8554/stream</Value>\r\n"
                            "\t\t</Setting>\r\n"
                            "\t\t<Setting Name=\"UCPCDVideoSource\" Type=\"System.String\" Scope=\"User\">\r\n"
                              "\t\t\t<Value Profile=\"(Default)\">https://192.168.43.173:3000?fcs=true</Value>\r\n"
                            "\t\t</Setting>\r\n"
                        "\t</Settings>\r\n"
                    "</SettingsFile>\r\n";
        FileController::addLine(fileName.toStdString(),text);
    }else if(fileName.toLower().contains("pcs.conf")){
        std::string text =
                    "<?xml version='1.0' encoding='utf-8'?>\r\n"
                    "<SettingsFile>\r\n"
                      "\t<Settings>\r\n"
                        "\t\t<Setting Name=\"LinkName\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">GREMSY</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"LinkType\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">GREMSY</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"GimbalType\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">GREMSY</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"GimbalIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">10.42.0.230</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"GimbalPortIn\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">18001</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"GimbalPortOut\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">18002</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"SensorIP\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">192.168.0.103</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"SensorPortIn\" Type=\"System.Int32\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">11999</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"StreamEO\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">udpsrc port=8600 ! tsdemux ! tee name=t t. ! queue ! h265parse ! nvh265dec ! videoconvert ! video/x-raw,format=I420 </Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"StreamIR\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">filesrc location=rtspsrc location=rtsp://192.168.0.103/z3-2.sdp latency=100 do-retransmission=true drop-on-latency=true ! rtph265depay ! tee name=t t. ! queue ! h265parse ! nvh265dec ! videoconvert ! video/x-raw,format=I420 ! videoscale ! video/x-raw,width=1920,height=1080 </Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"MapDefault\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">Layers.tpk</Value>\r\n"
                        "\t\t</Setting>\r\n"
                        "\t\t<Setting Name=\"ElevationDataPath\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">ElevationData-H1</Value>\r\n"
                        "\t\t</Setting>\r\n"
                      "\t</Settings>\r\n"
                    "</SettingsFile>\r\n";
        FileController::addLine(fileName.toStdString(),text);
    }else if(fileName.toLower().contains("app.conf")){
        std::string text =
                    "<?xml version='1.0' encoding='utf-8'?>\r\n"
                    "<SettingsFile>\r\n"
                      "\t<Settings>\r\n"
                        "\t\t<Setting Name=\"Language\" Type=\"System.String\" Scope=\"User\">\r\n"
                          "\t\t\t<Value Profile=\"(Default)\">EN</Value>\r\n"
                        "\t\t</Setting>\r\n"
                      "\t</Settings>\r\n"
                    "</SettingsFile>\r\n";
        FileController::addLine(fileName.toStdString(),text);
    }
}

void Config::setPropertyValue(QString name,QString value){
    if(name == ""){
        return;
    }
    for(int i=0; i< m_paramsModel.size(); i++){
        if(m_paramsModel[i]->name().toUpper() == name.toUpper()){
            m_paramsModel[i]->setValue(value);
            changeData(name,value);
            break;
        }
    }
}
