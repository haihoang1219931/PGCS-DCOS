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
public:
    void connectToGimbal(Config* config = nullptr) override;
    void disconnectGimbal() override;
Q_SIGNALS:

public Q_SLOTS:
private:
    GRGimbalController* m_gimbal;
};

#endif // GREMSEYGIMBAL_H
