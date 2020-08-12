#ifndef COMCONTROLLER_H
#define COMCONTROLLER_H

#include <QObject>
#include <QThread>
#include "ComTCP.h"
class ComController : public QObject
{
    Q_OBJECT
public:
    explicit ComController(QObject *parent = nullptr);
    virtual ~ComController();
Q_SIGNALS:

public Q_SLOTS:
    void setAddress(QString ip, int port);
    void start();
    void stop();
public:
    ComTCP* m_comTCP = nullptr;
    QThread* m_comTCPThread = nullptr;
};

#endif // COMCONTROLLER_H
