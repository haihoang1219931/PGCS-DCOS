#ifndef JOYSTICKTHREAD_H
#define JOYSTICKTHREAD_H

#include <QObject>
#include <QVector>
#include <QString>
#include <QVariant>
#include <QVariantMap>
#include <QStringList>
#include <QThread>
#include <QQmlApplicationEngine>
#include <QQmlListProperty>
#include <stdio.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

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
public:
    JSAxis(QObject *parent = nullptr){ Q_UNUSED(parent);}
    JSAxis(int id, float value = 0){ m_id =id; m_value = value;}
    int id(){ return m_id;}
    void setId(int id){m_id = id; Q_EMIT idChanged();}
    float value(){ return m_value;}
    void setValue(float value){m_value = value; Q_EMIT valueChanged();}
Q_SIGNALS:
    void idChanged();
    void valueChanged();
public Q_SLOTS:
private:
    int m_id;
    float m_value;
};
class JSButton: public QObject
{
    Q_OBJECT
    Q_PROPERTY(int id READ id WRITE setId NOTIFY idChanged)
    Q_PROPERTY(bool pressed READ pressed WRITE setPressed NOTIFY pressedChanged)
public:
    JSButton(QObject *parent = nullptr){ Q_UNUSED(parent);}
    JSButton(int id, bool pressed = false){ m_id =id; m_pressed = pressed;}
    int id(){ return m_id;}
    void setId(int id){m_id = id; Q_EMIT idChanged();}
    bool pressed(){ return m_pressed;}
    void setPressed(bool pressed){m_pressed = pressed; Q_EMIT pressedChanged();}
Q_SIGNALS:
    void idChanged();
    void pressedChanged();
public Q_SLOTS:
private:
    int m_id;
    bool m_pressed;
};

class JoystickThreaded: public QObject
{
    Q_OBJECT
    Q_PROPERTY(JoystickTask* task READ task)
    Q_PROPERTY(QQmlListProperty<JSAxis> axes READ axes NOTIFY axesChanged)
    Q_PROPERTY(QQmlListProperty<JSButton> buttons READ buttons NOTIFY buttonsChanged)
public:
    JoystickThreaded(QObject *parent = nullptr);

    virtual ~JoystickThreaded();
    Q_INVOKABLE void start();
    Q_INVOKABLE void pause(bool pause);
    Q_INVOKABLE void stop();
    Q_INVOKABLE void setJoyID(QString joyID);
    JoystickTask* task(){
        return m_task;
    }
    // List of Axis
    QQmlListProperty<JSAxis> axes(){
        return QQmlListProperty<JSAxis>(this, this,
                                                &JoystickThreaded::appendAxis,
                                                &JoystickThreaded::axisCount,
                                                &JoystickThreaded::axis,
                                                &JoystickThreaded::clearAxes);
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
    // List of Button
    QQmlListProperty<JSButton> buttons(){
        return QQmlListProperty<JSButton>(this, this,
                                                &JoystickThreaded::appendButton,
                                                &JoystickThreaded::buttonCount,
                                                &JoystickThreaded::button,
                                                &JoystickThreaded::clearButtons);
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
    static void expose(){
        qmlRegisterType<JSAxis>();
        qmlRegisterType<JSButton>();
        qmlRegisterType<JoystickThreaded>("io.qdt.dev", 1, 0, "Joystick");
    }
public Q_SLOTS:
    QStringList getListJoystick();
    QVariant getJoystickInfo(QString jsFile);
    void updateButtonAxis(bool connected);
    void btnClicked(int btnID,bool clicked);
    void axisStateChanged(int axisID, float value);
Q_SIGNALS:
    void joystickConnected(bool state);
    void axesChanged();
    void buttonsChanged();
    void axisValueChanged(int axisID, float value);
private:
    QThread *m_workerThread;
    JoystickTask* m_task = NULL;
    QVector<JSAxis*> m_axes;
    QVector<JSButton*> m_buttons;
private:
    static void appendAxis(QQmlListProperty<JSAxis>* list, JSAxis* p) {
        reinterpret_cast<JoystickThreaded* >(list->data)->appendAxis(p);
    }
    static void clearAxes(QQmlListProperty<JSAxis>* list) {
        reinterpret_cast<JoystickThreaded* >(list->data)->clearAxes();
    }
    static JSAxis* axis(QQmlListProperty<JSAxis>* list, int i) {
        return reinterpret_cast<JoystickThreaded* >(list->data)->axis(i);
    }
    static int axisCount(QQmlListProperty<JSAxis>* list) {
        return reinterpret_cast<JoystickThreaded* >(list->data)->axisCount();
    }

    static void appendButton(QQmlListProperty<JSButton>* list, JSButton* p) {
        reinterpret_cast<JoystickThreaded* >(list->data)->appendButton(p);
    }
    static void clearButtons(QQmlListProperty<JSButton>* list) {
        reinterpret_cast<JoystickThreaded* >(list->data)->clearButtons();
    }
    static JSButton* button(QQmlListProperty<JSButton>* list, int i) {
        return reinterpret_cast<JoystickThreaded* >(list->data)->button(i);
    }
    static int buttonCount(QQmlListProperty<JSButton>* list) {
        return reinterpret_cast<JoystickThreaded* >(list->data)->buttonCount();
    }

};

#endif // JOYSTICKTHREAD_H
