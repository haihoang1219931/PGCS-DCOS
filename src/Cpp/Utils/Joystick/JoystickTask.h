#ifndef JOYSTICKTASK_H
#define JOYSTICKTASK_H

#ifdef KFJOYSTICK
    #include "../ControllerLib/Buffer/RollBuffer.h"
    #include "KFJoystick.h"
#endif
#include "JoystickController.h"
#include <QMutex>
#include <QObject>
#include <QWaitCondition>
#include <QStringList>
#include <QVariant>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
class JoystickTask : public QObject
{
        Q_OBJECT
        Q_PROPERTY(bool stop READ stop WRITE setStop)
        Q_PROPERTY(QString joyID READ joyID WRITE setJoyID NOTIFY joyIDChanged)

    public:
        explicit JoystickTask(QObject* parent = 0);
        ~JoystickTask();
        Q_INVOKABLE void setJoyID(QString a);
        QString joyID();
        void setStop(bool a);
        bool stop();

    Q_SIGNALS:
        void btnClicked(int btnID, bool clicked);
        void joystickConnected(bool state);
        void joyIDChanged();
        void axisStateChanged(int axisID, float value);
    public Q_SLOTS:
        void pause(bool _pause);
        void doWork();
        QStringList getListJoystick();
        QVariant getJoystickInfo(QString jsFile);
    public:
        Joystick m_joystick;
        QString m_joyID = "/dev/input/js0";
        float m_pan;
        float m_tilt;
        int m_zoom;
        bool m_stop = false;
        bool m_pause = false;
        QMutex m_mutexProcess;
        QWaitCondition m_pauseCond;
#ifdef KFJOYSTICK
        KFJoystick m_kf;

        RollBuffer<float> m_rollPan;
        RollBuffer<float> m_rollTilt;
#endif
};

#endif // JOYSTICKTASK_H
