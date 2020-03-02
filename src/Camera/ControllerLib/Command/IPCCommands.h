#ifndef IPCCOMMANDS_H
#define IPCCOMMANDS_H

#include "../Buffer/BufferOut.h"
#include "../Packet/EyeStatus.h"
#include "../Packet/EyephoenixProtocol.h"
#include "../Packet/GimbalMode.h"
#include "../Packet/GimbalRecord.h"
#include "../Packet/HConfigMessage.h"
#include "../Packet/ImageStab.h"
#include "../Packet/KLV.h"
#include "../Packet/LockMode.h"
#include "../Packet/RFData.h"
#include "../Packet/RFRequest.h"
#include "../Packet/SceneSteering.h"
#include "../Packet/ScreenPoint.h"
#include "../Packet/SensorColor.h"
#include "../Packet/SensorId.h"
#include "../Packet/Snapshot.h"
#include "../Packet/StreamingProfile.h"
#include "../Packet/TrackObject.h"
#include "../Packet/TrackSize.h"
#include "../Packet/ZoomData.h"
#include "../Packet/ZoomStatus.h"
#include "../gimbalinterfacecontext.h"
#include <QQuickItem>
#include <QSocketNotifier>
#include <QUdpSocket>
#include "../tcp/clientStuff.h"
#include "../tcp/gimbal_control.h"
class IPCCommands : public QObject {
  Q_OBJECT
public:
  IPCCommands(QObject *parent = 0);
  virtual ~IPCCommands();

public:
  Q_INVOKABLE void changeTrackSize(int size);
  Q_INVOKABLE void changeStreamProfile(QString profile);
  Q_INVOKABLE void changeSensorID(QString sensorID);
  Q_INVOKABLE void changeLockMode(QString mode, QString geoLocation);
  Q_INVOKABLE void takeSnapshot(QString mode, int frameID);
  Q_INVOKABLE void changeSensorColor(QString mode);
  Q_INVOKABLE void takeRFRequest(QString mode, int frameID);
  Q_INVOKABLE void changeGimbalMode(QString mode);
  Q_INVOKABLE void changeRecordMode(QString mode, int type, int frameID);
  Q_INVOKABLE void setClickPoint(int frameID, double x, double y, double width,
                                 double height, double objW, double objH);
  Q_INVOKABLE void setZoomPosition(QString mode, int factor = 0);
  Q_INVOKABLE void enableImageStab(QString mode, double crop);
  Q_INVOKABLE void doSceneSteering(int _index);
  Q_INVOKABLE void configCamera(QString cmdID, QString cmdData);

  // control rtr camera
  Q_INVOKABLE void treronChangeSensor(QString sensorID);
  Q_INVOKABLE void treronZoomIn();
  Q_INVOKABLE void treronZoomOut();
  Q_INVOKABLE void treronZoomStop();
  Q_INVOKABLE void treronGetZoomPos();
  // control rtr gimbal
  Q_INVOKABLE void gimbalControl(int _roll, int _pan, int _tilt);

public:
  BufferOut *m_buffer;
  GimbalInterfaceContext *m_gimbalModel;
  ClientStuff *m_tcpSensor = nullptr;
  ClientStuff *m_tcpGimbal = nullptr;
};

#endif
