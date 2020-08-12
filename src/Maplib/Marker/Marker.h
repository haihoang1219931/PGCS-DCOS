#ifndef MARKER_H
#define MARKER_H
#include <iostream>
#include <QObject>
#include <QString>
using namespace std;
class Marker : public QObject{
    Q_OBJECT

    Q_PROPERTY(QString latitude READ latitude WRITE setLatitude)
    Q_PROPERTY(QString longtitude READ longtitude WRITE setLongtitude)
    Q_PROPERTY(QString description READ description WRITE setDescription)
    Q_PROPERTY(QString elevation READ elevation WRITE setElevation)
    Q_PROPERTY(QString markerType READ markerType WRITE setMarkerType)
public:
    explicit Marker(QObject *parent = nullptr);
    virtual ~Marker();
public:
    QString latitude(){ return QString::fromStdString(m_Latitude);}
    QString longtitude(){ return QString::fromStdString(m_Longtitude);}
    QString description(){ return QString::fromStdString(m_Description);}
    QString elevation(){ return QString::fromStdString(m_Elevation);}
    QString markerType(){ return QString::fromStdString(m_MarkerType);}

    void setLatitude(QString _latitude){
        m_Latitude = _latitude.toStdString();
    }
    void setLongtitude(QString _longtitude){
        m_Longtitude = _longtitude.toStdString();
    }
    void setDescription(QString _description){
        m_Description = _description.toStdString();
    }
    void setElevation(QString _elevation){
        m_Elevation = _elevation.toStdString();
    }
    void setMarkerType(QString _markerType){
        m_MarkerType = _markerType.toStdString();
    }

public:
    string m_Latitude;
    string m_Longtitude;
    string m_Description;
    string m_Elevation;
    string m_MarkerType;
};

#endif
