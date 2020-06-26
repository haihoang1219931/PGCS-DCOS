#ifndef GEOCOMMAND_H
#define GEOCOMMAND_H

#include <QQuickItem>
#include <QSocketNotifier>
#include <QUdpSocket>

#include "../Packet/EyephoenixProtocol.h"
#include "../Packet/KLV.h"
#include "../Packet/GPSData.h"
#include "../Packet/RequestResponsePacket.h"
#include "Camera/Buffer/BufferOut.h"
#include "../../GimbalData.h"
#include "../Packet/EnGeoLocation.h"

class GeoCommands: public QObject
{
    Q_OBJECT
public:
    GeoCommands(QObject* parent = 0);
    virtual ~GeoCommands();
public:
    Q_INVOKABLE void sendGPSData(double pn, double pe, double pd);
    Q_INVOKABLE void sendGeoLocation();
public:
    BufferOut *m_buffer;
    GimbalData* m_gimbalModel;
};

#endif // GEOCOMMAND_H
