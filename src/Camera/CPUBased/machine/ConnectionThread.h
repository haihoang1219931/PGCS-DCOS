#ifndef CONNECTIONTHREAD_H
#define CONNECTIONTHREAD_H

#include <QObject>
#include <QThread>
#include "ConnectionChecking.h"
class ConnectionThread : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString address READ address WRITE setAddress)
public:
    explicit ConnectionThread(QObject *parent = nullptr);
    virtual ~ConnectionThread();
public:
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
    QString address(){
        return QString::fromStdString(m_connection->m_address);
    }
    void setAddress(QString address){
        m_connection->m_address = address.toStdString();
    }
Q_SIGNALS:
    void stateChanged(QString state);
public Q_SLOTS:
    void changeState(QString state);
public:
    QThread* m_workerThread;
    ConnectionChecking * m_connection = NULL;

};

#endif // CONNECTIONTHREAD_H
