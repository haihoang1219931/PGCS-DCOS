#ifndef JOYSTICKTHREAD_H
#define JOYSTICKTHREAD_H

#include <QObject>
#include <QVector>
#include <QString>
#include <QVariant>
#include <QVariantMap>
#include <QStringList>
#include <QThread>
#include <QList>
#include <QQmlApplicationEngine>
#include <QQmlListProperty>
#include <stdio.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../../Setting/tinyxml2.h"
#ifndef FILENAME_MAX
    #define FILENAME_MAX 100
#endif
#include "JoystickTask.h"

/**/
class JSAxis: public QObject
{
    Q_OBJECT
    Q_PROPERTY(int id READ id WRITE setId NOTIFY idChanged)    
    Q_PROPERTY(float value READ value WRITE setValue NOTIFY valueChanged)
    Q_PROPERTY(bool inverted READ inverted WRITE setInverted NOTIFY invertedChanged)
    Q_PROPERTY(QString mapFunc READ mapFunc WRITE setMapFunc NOTIFY mapFuncChanged)
    Q_PROPERTY(bool invertedConfig READ invertedConfig WRITE setInvertedConfig NOTIFY invertedConfigChanged)
    Q_PROPERTY(QString mapFuncConfig READ mapFuncConfig WRITE setMapFuncConfig NOTIFY mapFuncConfigChanged)
public:
    JSAxis(QObject *parent = nullptr){ Q_UNUSED(parent);}
    JSAxis(int id, float value = 0){ m_id =id; m_value = value;}
    int id(){ return m_id;}
    void setId(int id){m_id = id; Q_EMIT idChanged();}
    float value(){ return m_value;}
    void setValue(float value){m_value = value; Q_EMIT valueChanged();}
    QString mapFunc(){ return m_mapFunc; }
    void setMapFunc(QString mapFunc){
        m_mapFunc = mapFunc;
        Q_EMIT mapFuncChanged();
        m_mapFuncConfig = mapFunc;
        Q_EMIT mapFuncConfigChanged();
    }
    QString mapFuncConfig(){ return m_mapFuncConfig; }
    void setMapFuncConfig(QString mapFuncConfig){
        m_mapFuncConfig = mapFuncConfig;
        Q_EMIT mapFuncConfigChanged();
    }
    bool inverted(){return m_inverted;}
    void setInverted(bool invert){
        if(m_inverted!=invert){
            m_inverted = invert;
            Q_EMIT invertedChanged();
        }
        if(m_invertedConfig!=invert){
            m_invertedConfig = invert;
            Q_EMIT invertedConfigChanged();
        }
    }
    bool invertedConfig(){return m_invertedConfig;}
    void setInvertedConfig(bool invertConfig){
        if(m_invertedConfig!=invertConfig){
            m_invertedConfig = invertConfig;
            Q_EMIT invertedConfigChanged();
        }
    }
    void saveConfig(){
        m_inverted = m_invertedConfig;
        m_mapFunc = m_mapFuncConfig;
    }
Q_SIGNALS:
    void idChanged();
    void valueChanged();
    void invertedChanged();
    void invertedConfigChanged();
    void mapFuncChanged();
    void mapFuncConfigChanged();

public Q_SLOTS:
private:
    int m_id = 0;
    float m_value = 0;
    bool m_inverted = false;
    bool m_invertedConfig = false;
    QString m_mapFunc = "Unused";
    QString m_mapFuncConfig = "Unused";
};
class JSButton: public QObject
{
    Q_OBJECT
    Q_PROPERTY(int id READ id WRITE setId NOTIFY idChanged)
    Q_PROPERTY(bool pressed READ pressed WRITE setPressed NOTIFY pressedChanged)
    Q_PROPERTY(QString mapFunc READ mapFunc WRITE setMapFunc NOTIFY mapFuncChanged)
    Q_PROPERTY(QString mapFuncConfig READ mapFuncConfig WRITE setMapFuncConfig NOTIFY mapFuncConfigChanged)
public:
    JSButton(QObject *parent = nullptr){ Q_UNUSED(parent);}
    JSButton(int id, bool pressed = false){ m_id =id; m_pressed = pressed;}
    int id(){ return m_id;}
    void setId(int id){m_id = id; Q_EMIT idChanged();}
    bool pressed(){ return m_pressed;}
    void setPressed(bool pressed){m_pressed = pressed; Q_EMIT pressedChanged();}
    QString mapFunc(){ return m_mapFunc; }
    void setMapFunc(QString mapFunc){
        m_mapFunc = mapFunc;
        Q_EMIT mapFuncChanged();
    }
    QString mapFuncConfig(){ return m_mapFuncConfig; }
    void setMapFuncConfig(QString mapFuncConfig){
        m_mapFuncConfig = mapFuncConfig;
        Q_EMIT mapFuncConfigChanged();
    }
    void saveConfig(){
        m_mapFunc = m_mapFuncConfig;
    }
Q_SIGNALS:
    void idChanged();
    void pressedChanged();
    void mapFuncChanged();
    void mapFuncConfigChanged();
public Q_SLOTS:
private:
    int m_id = 0;
    bool m_pressed = false;
    QString m_mapFunc = "Unused";
    QString m_mapFuncConfig = "Unused";
};

class JoystickThreaded: public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString mapFile READ mapFile WRITE setMapFile)
    Q_PROPERTY(JoystickTask* task READ task)
    Q_PROPERTY(bool connected READ connected WRITE setConnected NOTIFY joystickConnected)
    Q_PROPERTY(bool                 useJoystick                 READ useJoystick        WRITE setUseJoystick        NOTIFY useJoystickChanged)
    Q_PROPERTY(bool                 pic                         READ pic                WRITE setPIC                NOTIFY picChanged)
    Q_PROPERTY(QQmlListProperty<JSAxis> axes READ axes NOTIFY axesChanged)
    Q_PROPERTY(QQmlListProperty<JSAxis> axesCam READ axesCam NOTIFY axesCamChanged)
    Q_PROPERTY(QQmlListProperty<JSButton> buttons READ buttons NOTIFY buttonsChanged)

public:
    JoystickThreaded(QObject *parent = nullptr);

    virtual ~JoystickThreaded();
    Q_INVOKABLE void start();
    Q_INVOKABLE void pause(bool pause);
    Q_INVOKABLE void stop();
    Q_INVOKABLE void setJoyID(QString joyID);
    Q_INVOKABLE void saveConfig();
    Q_INVOKABLE void loadConfig();
    Q_INVOKABLE void resetConfig();
    Q_INVOKABLE void mapAxis(int axisID, QString mapFunc, bool invert, bool saveCurrent = true, bool axisCam = false);
    Q_INVOKABLE void mapButton(int buttonID, QString mapFunc, bool saveCurrent = true);
    Q_INVOKABLE void setInvertCam(QString camFunc,bool invert);
    QString mapFile(){ return m_mapFile; }
    void setMapFile(QString mapFile){
        m_mapFile = mapFile;
    }
    JoystickTask* task(){
        return m_task;
    }
    // List of Axis
    QQmlListProperty<JSAxis> axes(){
        return QQmlListProperty<JSAxis>(this, m_axes);
    }
    void appendAxis(JSAxis* p) {
        m_axes.append(p);
        Q_EMIT axesChanged();
    }
    int axisCount() const{return m_axes.count();}
    JSAxis *axis(int index) const{ return m_axes.at(index);}
    void clearAxes() {
        m_axes.clear();
        Q_EMIT axesChanged();
    }
    // List of AxisCam
    QQmlListProperty<JSAxis> axesCam(){
        return QQmlListProperty<JSAxis>(this, m_axesCam);
    }
    void appendAxisCam(JSAxis* p) {
        m_axesCam.append(p);
        Q_EMIT axesCamChanged();
    }
    int axisCamCount() const{return m_axesCam.count();}
    JSAxis *axisCam(int index) const{ return m_axesCam.at(index);}
    void clearAxesCam() {
        m_axesCam.clear();
        Q_EMIT axesCamChanged();
    }
    // List of Button
    QQmlListProperty<JSButton> buttons(){
        return QQmlListProperty<JSButton>(this, m_buttons);
    }
    void appendButton(JSButton* p) {
        m_buttons.append(p);
        Q_EMIT buttonsChanged();
    }
    int buttonCount() const{return m_buttons.count();}
    JSButton *button(int index) const{ return m_buttons.at(index);}
    void clearButtons() {
        m_buttons.clear();
        Q_EMIT buttonsChanged();
    }

    int axisRoll(){ return m_axisRoll; }
    int axisPitch(){ return m_axisPitch; }
    int axisYaw(){ return m_axisYaw; }
    int axisThrottle(){ return m_axisThrottle; }

    void setAxisRoll(int value){
        if(m_axisRoll != value){
            m_axisRoll = value;
        }
    }
    void setAxisPitch(int value){
        if(m_axisPitch != value){
            m_axisPitch = value;
        }
    }
    void setAxisYaw(int value){
        if(m_axisYaw != value){
            m_axisYaw = value;
        }
    }
    void setAxisThrottle(int value){
        if(m_axisThrottle != value){
            m_axisThrottle = value;
        }
    }

    static void expose(){
        qmlRegisterType<JSAxis>();
        qmlRegisterType<JSButton>();
        qmlRegisterType<JoystickThreaded>("io.qdt.dev", 1, 0, "Joystick");
    }
    bool connected(){ return m_connected; }

    bool pic(){return m_pic;}
    void setPIC(bool pic){
        m_pic = pic;
        printf("%s = %s\r\n",__func__,m_pic?"true":"false");
        Q_EMIT picChanged();
    }
    bool useJoystick(){return m_useJoystick;}
    void setUseJoystick(bool enable){        
        m_useJoystick = enable;
        printf("%s = %s\r\n",__func__,m_useJoystick?"true":"false");
        Q_EMIT useJoystickChanged(m_useJoystick);
    }
    int axisPan(){ return m_axisPan; }
    int axisTilt(){ return m_axisTilt; }
    int axisZoom(){ return m_axisZoom; }
    int invertPan(){
        if(m_axisPan >=0 && m_axisPan < m_axesCam.size()){
            return m_axesCam[m_axisPan]->inverted()?-1:1;
        }else{
            return 1;
        }
    }
    int invertTilt(){
        if(m_axisTilt >=0 && m_axisTilt < m_axesCam.size()){
            return m_axesCam[m_axisTilt]->inverted()?-1:1;
        }else{
            return 1;
        }
    }
    int invertZoom(){
        if(m_axisZoom >=0 && m_axisZoom < m_axesCam.size()){
            return m_axesCam[m_axisZoom]->inverted()?-1:1;
        }else{
            return 1;
        }
    }
public Q_SLOTS:
    void updateButtonAxis(bool connected);
    void changeButtonState(int btnID,bool clicked);
    void changeAxisValue(int axisID, float value);
    void setConnected(bool connected){
        printf("Joystick connected = %s\r\n",connected?"true":"false");
        m_connected = connected;
        Q_EMIT joystickConnected(m_connected);
    }
Q_SIGNALS:
    void useJoystickChanged(bool useJoystick);
    void picChanged();
    void buttonAxisLoaded();
    void joystickConnected(bool state);
    void axesChanged();
    void axesCamChanged();
    void buttonsChanged();
    void axisValueChanged(int axisID, float value);
    void buttonStateChanged(int buttonID, bool pressed);
private:
    QThread *m_workerThread;
    JoystickTask* m_task = nullptr;
    QList<JSAxis*> m_axes;
    QList<JSAxis*> m_axesCam;
    QList<JSButton*> m_buttons;
    QString m_mapFile;
    bool m_connected = false;
    int m_axisRoll = 0;
    int m_axisPitch = 1;
    int m_axisYaw = 3;
    int m_axisThrottle = 2;
    int m_axisPan = 0;
    int m_axisTilt = 1;
    int m_axisZoom = 2;
    int m_butonPICCIC = 0;
    bool m_pic = false;
    bool m_useJoystick = false;
};

#endif // JOYSTICKTHREAD_H
