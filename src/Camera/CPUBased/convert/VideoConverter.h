#ifndef VIDEOCONVERTER_H
#define VIDEOCONVERTER_H

#include <QObject>
#include <QMutex>
#include <QWaitCondition>
class VideoConverter: public QObject
{
    Q_OBJECT
public:
    explicit VideoConverter(QObject *parent = nullptr);
    virtual ~VideoConverter();
public:
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
    Q_INVOKABLE void setState(QString state);
Q_SIGNALS:
    void stateChange(QString state);
public Q_SLOTS:
    void doWork();
public:
    QMutex *m_mutex;
    QWaitCondition *m_pauseCond;
    bool m_stop = true;
};

#endif // VIDEOCONVERTER_H
