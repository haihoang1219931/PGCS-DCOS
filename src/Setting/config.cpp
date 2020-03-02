#include "config.h"

Config::Config(QObject *parent) : QObject(parent)
{
    //ctor
}

int Config::changeData(QString data,QString value){
    Q_UNUSED(data);
    Q_UNUSED(value);
    return 0;
}

void Config::print(){
}
int Config::readConfig(QString file){
    mapData.clear();
    if(m_doc.LoadFile(file.toStdString().c_str()) != tinyxml2::XML_SUCCESS){
        printf("Failed to load file\r\n");
        return -1;
    }
    XMLElement* settings = m_doc.FirstChildElement("SettingsFile")->FirstChildElement("Settings");
    int countSetting = 0;
    if(settings != nullptr){
        QVariantMap settingsMap;
        for(tinyxml2::XMLElement* child = settings->FirstChildElement(); child != nullptr; child = child->NextSiblingElement()){
            QVariantMap settingMap;
            countSetting ++;
            QVariantMap attributesMap;
            attributesMap.insert("Name",QString::fromUtf8(child->Attribute("Name")));
            attributesMap.insert("Type",QString::fromUtf8(child->Attribute("Type")));
            attributesMap.insert("Scope",QString::fromUtf8(child->Attribute("Scope")));
            QVariantMap valueMap;
            valueMap.insert("data",QString::fromUtf8(child->FirstChildElement("Value")->GetText()));
            valueMap.insert("profile",QString::fromUtf8(child->Attribute("Profile")));
            attributesMap.insert("Name",QString::fromUtf8(child->Attribute("Name")));
            settingMap.insert("Attributes",attributesMap);
            settingMap.insert("Value",valueMap);
            settingsMap.insert(QString::fromUtf8(child->Attribute("Name")),settingMap);
        }
        mapData.insert("Settings",settingsMap);
    }else{
        printf("Failed to find element\r\n");
    }
    m_data = QVariant(mapData);
    print();
    return 0;
}
QVariant Config::getData(){
    return m_data;
}
int Config::saveConfig(QString file){
    m_doc.SaveFile(file.toStdString().c_str(),false);
    return 0;
}
QVariant Config::value(QString pattern){
    QVariant value;
    QStringList listKey = pattern.split(":");
    QVariantMap valueMap = mapData;
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
