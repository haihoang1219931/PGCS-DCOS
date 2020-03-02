#ifndef PROTOCOLINTERFACE_H
#define PROTOCOLINTERFACE_H

#include <QObject>

class ProtocolInterface : public QObject
{
    Q_OBJECT
public:
    explicit ProtocolInterface(QObject *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:
};

#endif // PROTOCOLINTERFACE_H
