#ifndef MESSAGEMANAGER_H
#define MESSAGEMANAGER_H

#include <QObject>
#include <QQueue>
#include <QList>
#include <mavlink.h>
#ifndef MAX_QUEUE_LENGTH
#define MAX_QUEUE_LENGTH 1000
#endif
class MessageManager : public QObject
{
    Q_OBJECT
public:
    enum QUEUE_ID{
        QUEUE_IN = 0,
        QUEUE_OUT = 0,
    };
    explicit MessageManager(QObject *parent = nullptr);
    virtual ~MessageManager();
Q_SIGNALS:

public Q_SLOTS:

public:
    void push(mavlink_message_t msg,QUEUE_ID queueID);
    QList<mavlink_message_t> pop(QUEUE_ID queueID);
public:
    QQueue<mavlink_message_t> m_queueIn;
    QQueue<mavlink_message_t> m_queueOut;
};

#endif // MESSAGEMANAGER_H
