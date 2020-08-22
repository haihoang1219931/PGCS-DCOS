#ifndef MAVLINKPROTOCOL_H
#define MAVLINKPROTOCOL_H

#include <QObject>

class MavlinkProtocol : public QObject
{
    Q_OBJECT
public:
    explicit MavlinkProtocol(QObject *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:
};

#endif // MAVLINKPROTOCOL_H
