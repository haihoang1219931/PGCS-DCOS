#include "JoystickTask.h"

JoystickTask::JoystickTask(QObject* parent) : QObject(parent)
{
    #ifdef KFJOYSTICK
    math::Matrix A(4, 4);
    A(0, 0) = 1.0;
    A(0, 1) = 0.0;
    A(0, 2) = 0.1;
    A(0, 3) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 1.0;
    A(1, 2) = 0.0;
    A(1, 3) = 0.1;
    A(2, 0) = 0.0;
    A(2, 1) = 0.0;
    A(2, 2) = 1.0;
    A(2, 3) = 0.0;
    A(3, 0) = 0.0;
    A(3, 1) = 0.0;
    A(3, 2) = 0.0;
    A(3, 3) = 1.0;
    math::Matrix H(2, 4);
    H(0, 0) = 1.0;
    H(0, 1) = 0.0;
    H(0, 2) = 0.0;
    H(0, 3) = 0.0;
    H(1, 0) = 0.0;
    H(1, 1) = 1.0;
    H(1, 2) = 0.0;
    H(1, 3) = 0.0;
    math::Matrix Q(4, 4);
    Q(0, 0) = 1.0;
    Q(0, 1) = 0.0;
    Q(0, 2) = 0.0;
    Q(0, 3) = 0.0;
    Q(1, 0) = 0.0;
    Q(1, 1) = 1.0;
    Q(1, 2) = 0.0;
    Q(1, 3) = 0.0;
    Q(2, 0) = 0.0;
    Q(2, 1) = 0.0;
    Q(2, 2) = 0.01;
    Q(2, 3) = 0.0;
    Q(3, 0) = 0.0;
    Q(3, 1) = 0.0;
    Q(3, 2) = 0.0;
    Q(3, 3) = 0.01;
    math::Matrix R(2, 2);
    R(0, 0) = 100;
    R(0, 1) = -20;
    R(1, 0) = -20;
    R(1, 1) = 100;
    math::Matrix P(4, 4);
    P(0, 0) = 1.0;
    P(0, 1) = 0.0;
    P(0, 2) = 0.0;
    P(0, 3) = 0.0;
    P(1, 0) = 0.0;
    P(1, 1) = 1.0;
    P(1, 2) = 0.0;
    P(1, 3) = 0.0;
    P(2, 0) = 0.0;
    P(2, 1) = 0.0;
    P(2, 2) = 1.0;
    P(2, 3) = 0.0;
    P(3, 0) = 0.0;
    P(3, 1) = 0.0;
    P(3, 2) = 0.0;
    P(3, 3) = 1.0;
    math::Vector X0(4);
    X0(0) = 0;
    X0(1) = 0, X0(2) = 0;
    X0(3) = 0;
    m_kf.init(4, 2, A, H, R, Q, P, X0);
    m_rollPan.setSize(10);
    m_rollTilt.setSize(10);
#endif
}
JoystickTask::~JoystickTask()
{
}
void JoystickTask::pause(bool _pause)
{
    if (_pause == true)
    {
        m_mutexProcess.lock();
        m_pause = true;
        m_mutexProcess.unlock();
    }
    else
    {
        m_mutexProcess.lock();
        m_pause = false;
        m_mutexProcess.unlock();
        m_pauseCond.wakeAll();
    }
}
void JoystickTask::setJoyID(QString a)
{
    if (a != m_joyID)
    {
        pause(true);
        m_joyID = a;
        m_joystick.openPath(m_joyID.toStdString());
        pause(false);
        Q_EMIT joyIDChanged();
    }
}
QString JoystickTask::joyID()
{
    return m_joyID;
}
void JoystickTask::setStop(bool a)
{
    if (a != m_stop)
    {
        m_stop = a;
    }
}
bool JoystickTask::stop()
{
    return this->m_stop;
}

void JoystickTask::doWork()
{
    m_joystick.openPath(m_joyID.toStdString());
    printf("Start getting data from joy %s\r\n", m_joyID.toStdString().c_str());
    // Ensure that it was found and that we can use it
    sleep(2);

    if (!m_joystick.isFound())
    {
        printf("First not found joystick\r\n");
        Q_EMIT joystickConnected(false);

        while (m_joystick.isExist() == false)
        {
            //            printf("isExist = false\r\n");
            sleep(1);

            if (m_joystick.isExist() == true)
            {
                m_joystick.closePath();
                m_joystick.openPath(m_joystick.m_devicePath);

                if (m_joystick.isFound())
                {
                    Q_EMIT joystickConnected(true);
                    printf("Joystick connected\r\n");
                    break;
                }
            }
        }
    }
    else
    {
        printf("\r\nFrist found joystick\r\n");
        Q_EMIT joystickConnected(true);
    }

    m_stop = false;
    while (m_stop == false)
    {
        m_mutexProcess.lock();
        if (m_pause)
        {
            m_pauseCond.wait(&m_mutexProcess);    // in this place, your thread will stop to execute until someone calls resume
        }
        m_mutexProcess.unlock();
        // Restrict rate
        usleep(16);
        // Attempt to sample an event from the joystick
        JoystickEvent event;

        if (m_joystick.sample(&event))
        {
            if (event.isButton())
            {
//                qDebug("Button %u is %s\n", event.number, event.value == 0 ? "up" : "down");
                Q_EMIT btnClicked(static_cast<int>(event.number), event.value != 0);
            }
            else if (event.isAxis())
            {
//                qDebug("axisStateChanged %u value to %f\n", event.number, static_cast<float>(event.value));
                Q_EMIT axisStateChanged(event.number,static_cast<float>(event.value));
            }
        }
        else
        {
            if (m_joystick.isExist() == false)
            {
                Q_EMIT joystickConnected(false);
            }

            while (m_joystick.isExist() == false)
            {
                if (m_stop == true)
                {
                    break;
                }

                sleep(1);

                if (m_joystick.isExist() == true)
                {
                    m_joystick.closePath();
                    m_joystick.openPath(m_joystick.m_devicePath);

                    if (m_joystick.isFound())
                    {
                        Q_EMIT joystickConnected(true);
                        break;
                    }
                }
            }
        }
    }

    printf("Stop Joystick %s\r\n", m_joyID.toStdString().c_str());
}

