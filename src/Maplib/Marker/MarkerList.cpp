#include "MarkerList.h"

MarkerList::MarkerList(QObject *parent) : QObject(parent)
{

}
void MarkerList::cleanMarker(){
    m_listmarker.clear();
}
void MarkerList::insertMarker(QString lat,QString lon, QString type, QString description ){
    Marker* _marker = new Marker(parent());
    _marker->m_Latitude = lat.toStdString();
    _marker->m_Longtitude = lon.toStdString();
    _marker->m_Description = description.toStdString();
    _marker->m_MarkerType = type.toStdString();
    /*
    qDebug("insert (%s,%s):%s\r\n",
           lat.toStdString().c_str(),
           lat.toStdString().c_str(),
           type.toStdString().c_str());
    */
    m_listmarker.push_back(_marker);
}

void MarkerList::insertMarker(Marker* _marker){
    m_listmarker.push_back(_marker);
}
void MarkerList::saveMarkers(QString _fileName){
    XMLDocument xmlDoc;
    XMLNode * pRoot = xmlDoc.NewElement("ArrayOfMarkerBase");
    xmlDoc.InsertFirstChild(pRoot);
    for(unsigned int i=0; i< m_listmarker.size(); i++){
        Marker *tmp = m_listmarker[i];
        XMLElement * pElement = xmlDoc.NewElement("MarkerBase");
        XMLElement * pLat = xmlDoc.NewElement("Latitude");
        pLat->SetText(tmp->m_Latitude.c_str());
        pElement->InsertEndChild(pLat);
        XMLElement * pLon = xmlDoc.NewElement("Longitude");
        pLon->SetText(tmp->m_Longtitude.c_str());
        pElement->InsertEndChild(pLon);
        XMLElement * pDes = xmlDoc.NewElement("Description");
        pDes->SetText(tmp->m_Description.c_str());
        pElement->InsertEndChild(pDes);
        XMLElement * pEle = xmlDoc.NewElement("Elevation");
        pEle->SetText(tmp->m_Elevation.c_str());
        pElement->InsertEndChild(pEle);
        XMLElement * pTyp = xmlDoc.NewElement("MarkerType");
        pTyp->SetText(tmp->m_MarkerType.c_str());
        pElement->InsertEndChild(pTyp);
        pRoot->InsertEndChild(pElement);
    }

    XMLError eResult = xmlDoc.SaveFile(_fileName.toStdString().c_str());
}
void MarkerList::loadMarkers(QString _fileName){
    if(m_listmarker.size() != 0){
        m_listmarker.clear();
    }
    XMLDocument m_doc;
    XMLError res = m_doc.LoadFile(_fileName.toStdString().c_str());
    if(res != XML_SUCCESS){
        return;
    }
    XMLElement * pElement = m_doc.FirstChildElement("ArrayOfMarkerBase");
    XMLElement * pListElement = pElement->FirstChildElement("MarkerBase");
    //int i = 0;
    while(pListElement!= nullptr){
        //printf("item %d\r\n",i);
        Marker *tmp = new Marker(parent());
        XMLElement * pLat = pListElement->FirstChildElement("Latitude");
        XMLElement * pLon = pListElement->FirstChildElement("Longitude");
        XMLElement * pDes = pListElement->FirstChildElement("Description");
        XMLElement * pEle = pListElement->FirstChildElement("Elevation");
        XMLElement * pTyp = pListElement->FirstChildElement("MarkerType");
        /*
        printf("Latitude: %s\r\n",pLat->GetText());
        printf("Longitude: %s\r\n",pLon->GetText());
        printf("Elevation: %s\r\n",pEle->GetText());
        printf("Description: %s\r\n",pDes->GetText());
        printf("MarkerType: %s\r\n",pTyp->GetText());
        */
        if(pLat != nullptr) {
            if(pLat->GetText() != nullptr)
                tmp->m_Latitude = std::string(pLat->GetText());
            else{
                tmp->m_Latitude = "";
            }
        }
        if(pLon != nullptr) {
            if(pLon->GetText() != nullptr)
                tmp->m_Longtitude = std::string(pLon->GetText());
            else{
                tmp->m_Longtitude = "";
            }
        }
        if(pEle != nullptr) {
            if(pEle->GetText()!= nullptr)
                tmp->m_Elevation = std::string(pEle->GetText());
            else{
                tmp->m_Elevation = "";
            }
        }
        if(pDes != nullptr) {
            if(pDes->GetText()!= nullptr)
                tmp->m_Description = std::string(pDes->GetText());
            else{
                tmp->m_Description = "";
            }
        }
        if(pTyp != nullptr) {
            if(pTyp->GetText()!= nullptr)
                tmp->m_MarkerType = std::string(pTyp->GetText());
            else{
                tmp->m_MarkerType = "";
            }
        }

        pListElement = pListElement->NextSiblingElement("MarkerBase");
        //i++;
        m_listmarker.push_back(tmp);
    }
}
int MarkerList::numMarker(){
    return m_listmarker.size();
}
Marker* MarkerList::getMarker(int _markerID){
    return m_listmarker[_markerID];
}
