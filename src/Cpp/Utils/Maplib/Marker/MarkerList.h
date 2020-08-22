#ifndef MARKERLIST_H
#define MARKERLIST_H

#include <iostream>
#include <vector>
#include <QObject>

#include "Marker.h"
#include "../../Setting/tinyxml2.h"
using namespace std;
using namespace tinyxml2;
class MarkerList : public QObject
{
    Q_OBJECT
public:
    explicit MarkerList(QObject *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:

public:
    Q_INVOKABLE void cleanMarker();
    Q_INVOKABLE void insertMarker(Marker* _marker);
    Q_INVOKABLE void insertMarker(QString lat,QString lon, QString type, QString description = "");
    Q_INVOKABLE void saveMarkers(QString fileName);
    Q_INVOKABLE void loadMarkers(QString fileName);
    Q_INVOKABLE int numMarker();
    Q_INVOKABLE Marker* getMarker(int _markerID);
public:
    vector<Marker*> m_listmarker;
};

#endif // MARKERLIST_H
