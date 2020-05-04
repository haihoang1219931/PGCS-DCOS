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
    Q_PROPERTY(bool inverted READ inverted WRITE setInverted NOTIFY invertedChanged)
    Q_PROPERTY(float value READ value WRITE setValue NOTIFY valueChanged)
    Q_PROPERTY(QString mapFunc READ mapFunc WRITE setMapFunc NOTIFY mapFuncChanged)
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
    }
    bool inverted(){return m_inverted;}
    void setInverted(bool invert){
        if(m_inverted!=invert){
            m_inverted = invert;
            Q_EMIT invertedChanged();
        }
    }
Q_SIGNALS:
    void invertedChanged();
    void idChanged();
    void valueChanged();
    void mapFuncChanged();
public Q_SLOTS:
private:
    bool m_inverted = false;
    int m_id = 0;
    float m_value = 0;
    QString m_mapFunc = "Unused";
};
class JSButton: public QObject
{
    Q_OBJECT
    Q_PROPERTY(int id READ id WRITE setId NOTIFY idChanged)
    Q_PROPERTY(bool pressed READ pressed WRITE setPressed NOTIFY pressedChanged)
    Q_PROPERTY(QString mapFunc READ mapFunc WRITE setMapFunc NOTIFY mapFuncChanged)
public:
    JSButton(QObject *parent = nullptr){ Q_UNUSED(parent);}
    JSButton(int id, bool pressed = false){ m_id =id; m_pressed = pressed;}
    int id(){ return m_id;}
    void setId(int id){m_id = id; Q_EMIT idChanged();}
    bool pressed(){ return m_pressed;}
    void setPressed(bool pressed){m_pressed = pressed; Q_EMIT pressedChanged();}
    QString mapFunc(){ return m_mapFunc; }
    void setMapFunc(QString mapFunc){
        m_mapFunc = mapFunc; Q_EMIT mapFuncChanged();
    }
Q_SIGNALS:
    void idChanged();
    void pressedChanged();
    void mapFuncChanged();
public Q_SLOTS:
private:
    int m_id = 0;
    bool m_pressed = false;
    QString m_mapFunc = "Unused";
};

class JoystickThreaded: public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString mapFile READ mapFile WRITE setMapFile)
    Q_PROPERTY(JoystickTask* task READ task)
    Q_PROPERTY(QQmlListProperty<JSAxis> axes READ axes NOTIFY axesChanged)
    Q_PROPERTY(QQmlListProperty<JSButton> buttons READ buttons NOTIFY buttonsChanged)
    Q_PROPERTY(QQmlListProperty<JSAxis> axesConfig READ axesConfig NOTIFY axesConfigChanged)
    Q_PROPERTY(QQmlListProperty<JSButton> buttonsConfig READ buttonsConfig NOTIFY buttonsConfigChanged)
    Q_PROPERTY(int axisRoll READ axisRoll WRITE setAxisRoll NOTIFY axisRollChanged)
    Q_PROPERTY(int axisPitch READ axisPitch WRITE setAxisPitch NOTIFY axisPitchChanged)
    Q_PROPERTY(int axisYaw READ axisYaw WRITE setAxisYaw NOTIFY axisYawChanged)
    Q_PROPERTY(int axisThrottle READ axisThrottle WRITE setAxisThrottle NOTIFY axisThrottleChanged)

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
    Q_INVOKABLE void mapAxisConfig(int axisID, QString mapFunc, bool invert);
    Q_INVOKABLE void mapButtonConfig(int buttonID, QString mapFunc);
    Q_INVOKABLE void setInvert(QString camFunc,bool invert);
    void mapAxis(int axisID, QString mapFunc, bool invert);
    void mapButton(int buttonID, QString mapFunc);
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
    QQmlListProperty<JSAxis> axesConfig(){
        return QQmlListProperty<JSAxis>(this, m_axesTemp);
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
    JSButton *buttonTemp(int index) const{ return m_buttonsTemp.at(index);}
    void clearButtons() {
        m_buttons.clear();
        Q_EMIT buttonsChanged();
    }
    QQmlListProperty<JSButton> buttonsConfig(){
        return QQmlListProperty<JSButton>(this, m_buttonsTemp);
    }
    int axisRoll(){ return m_axisRoll; }
    int axisPitch(){ return m_axisPitch; }
    int axisYaw(){ return m_axisYaw; }
    int axisThrottle(){ return m_axisThrottle; }

    void setAxisRoll(int value){
        if(m_axisRoll != value){
            m_axisRoll = value;
            Q_EMIT axisRollChanged();
        }
    }
    void setAxisPitch(int value){
        if(m_axisPitch != value){
            m_axisPitch = value;
            Q_EMIT axisPitchChanged();
        }
    }
    void setAxisYaw(int value){
        if(m_axisYaw != value){
            m_axisYaw = value;
            Q_EMIT axisYawChanged();
        }
    }
    void setAxisThrottle(int value){
        if(m_axisThrottle != value){
            m_axisThrottle = value;
            Q_EMIT axisThrottleChanged();
        }
    }

    static void expose(){
        qmlRegisterType<JSAxis>();
        qmlRegisterType<JSButton>();
        qmlRegisterType<JoystickThreaded>("io.qdt.dev", 1, 0, "Joystick");
    }
    bool pic(){return m_pic;}
    void setPIC(bool pic){
        m_pic = pic;
        Q_EMIT picChanged();
    }
    bool useJoystick(){return m_useJoystick;}
    Q_INVOKABLE void setUseJoystick(bool enable){
        m_useJoystick = enable;
        Q_EMIT useJoystickChanged(m_useJoystick);
    }
    int axisPan(){ return m_axisPan; }
    int axisTilt(){ return m_axisTilt; }
    int axisZoom(){ return m_axisZoom; }
    int invertPan(){ return  m_invertPan; }
    int invertTilt(){ return m_invertTilt; }
    int invertZoom(){ return m_invertZoom; }
public Q_SLOTS:
    void updateButtonAxis(bool connected);
    void changeButtonState(int btnID,bool clicked);
    void changeAxisValue(int axisID, float value);
Q_SIGNALS:
    void useJoystickChanged(bool useJoystick);
    void picChanged();
    void buttonAxisLoaded();
    void joystickConnected(bool state);
    void axesChanged();
    void buttonsChanged();
    void axesConfigChanged();
    void buttonsConfigChanged();
    void axisValueChanged(int axisID, float value);
    void buttonStateChanged(int buttonID, bool pressed);
    void axisRollChanged();
    void axisPitchChanged();
    void axisYawChanged();
    void axisThrottleChanged();
private:
    QThread *m_workerThread;
    JoystickTask* m_task = nullptr;
    QList<JSAxis*> m_axes;
    QList<JSAxis*> m_axesTemp;
    QList<JSButton*> m_buttons;
    QList<JSButton*> m_buttonsTemp;
    QString m_mapFile;
    int m_axisRoll = 2;
    int m_axisPitch = 3;
    int m_axisYaw = 0;
    int m_axisThrottle = 1;
    int m_axisPan = 0;
    int m_axisTilt = 1;
    int m_axisZoom = 2;
    float m_invertPan = 1;
    float m_invertTilt = 1;
    float m_invertZoom = 1;
    int m_butonPICCIC = 0;
    bool m_pic = false;
    bool m_useJoystick = true;
};

#endif // JOYSTICKTHREAD_H
