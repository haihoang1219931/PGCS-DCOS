#ifndef GREMSEYGIMBAL_H
#define GREMSEYGIMBAL_H

#include <QObject>
#include "../GimbalInterface.h"
class GRGimbalController;
class GremseyGimbal : public GimbalInterface
{
    Q_OBJECT
public:
    explicit GremseyGimbal(GimbalInterface *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:
private:
    GRGimbalController* m_gimbal;
};

#endif // GREMSEYGIMBAL_H
