#ifndef MISSIONITEM_H
#define MISSIONITEM_H

#include <QObject>
#include <QList>
#include <QAbstractListModel>
#include <QGeoCoordinate>
#include "../Com/QGCMAVLink.h"
class MissionItem : public QObject
{
    Q_OBJECT

    Q_PROPERTY(int command READ command WRITE setCommand)
    Q_PROPERTY(int frame READ frame WRITE setFrame)
    Q_PROPERTY(int option READ option WRITE setOption)
    Q_PROPERTY(float param1 READ param1 WRITE setParam1)
    Q_PROPERTY(float param2 READ param2 WRITE setParam2)
    Q_PROPERTY(float param3 READ param3 WRITE setParam3)
    Q_PROPERTY(float param4 READ param4 WRITE setParam4)
    Q_PROPERTY(float param5 READ param5 WRITE setParam5)
    Q_PROPERTY(float param6 READ param6 WRITE setParam6)
    Q_PROPERTY(float param7 READ param7 WRITE setParam7)
    Q_PROPERTY(bool autoContinue READ autoContinue WRITE setAutoContinue)
    Q_PROPERTY(bool isCurrentItem READ isCurrentItem WRITE setIsCurrentItem)
    Q_PROPERTY(int sequence READ sequence WRITE setSequence)
    Q_PROPERTY(QGeoCoordinate position READ position WRITE setPosition)
public:
    explicit MissionItem(QObject *parent = nullptr);
    MissionItem(int             sequenceNumber,
                MAV_CMD         command,
                MAV_FRAME       frame,
                float           param1,
                float           param2,
                float           param3,
                float           param4,
                float           param5,
                float           param6,
                float           param7,
                bool            autoContinue,
                bool            isCurrentItem,
                QObject*        parent = NULL);
    int command(){ return static_cast<int>(m_command);}
    void setCommand(int command){ m_command = static_cast<MAV_CMD>(command);}
    int frame(){ return static_cast<int>(m_frame);}
    void setFrame(int frame){ m_frame = static_cast<MAV_FRAME>(frame);}
    int option(){ return m_option;}
    void setOption(int option){ m_option = option;}
    float param1(){return m_param1;}
    void setParam1(float value){m_param1 = value;}
    float param2(){return m_param2;}
    void setParam2(float value){m_param2 = value;}
    float param3(){return m_param3;}
    void setParam3(float value){m_param3 = value;}
    float param4(){return m_param4;}
    void setParam4(float value){m_param4 = value;}
    float param5(){return m_param5;}
    void setParam5(float value){m_param5 = value;}
    float param6(){return m_param6;}
    void setParam6(float value){m_param6 = value;}
    float param7(){return m_param7;}
    void setParam7(float value){m_param7 = value;}
    bool autoContinue(){ return m_autoContinue; }
    void setAutoContinue(bool value){ m_autoContinue = value; }
    bool isCurrentItem(){ return m_isCurrentItem; }
    void setIsCurrentItem(bool value) { m_isCurrentItem = value ;}
    int sequence(){return m_seq;}
    void setSequence(int value){m_seq = value;}
    QGeoCoordinate position(){ return QGeoCoordinate(static_cast<double>(m_param5),
                                                     static_cast<double>(m_param6),
                                                     static_cast<double>(m_param7));}
    void setPosition(QGeoCoordinate position){
        m_param5 = static_cast<float>(position.latitude());
        m_param6 =  static_cast<float>(position.longitude());
        m_param7 =  static_cast<float>(position.altitude());
    }
Q_SIGNALS:

public Q_SLOTS:
public:
    MAV_CMD     m_command;
    MAV_FRAME   m_frame;
    float      m_param1;
    float      m_param2;
    float      m_param3;
    float      m_param4;
    float      m_param5;
    float      m_param6;
    float      m_param7;
    bool        m_autoContinue;
    bool        m_isCurrentItem;
    int         m_option;
    int         m_seq;
};

#endif // MISSIONITEM_H
