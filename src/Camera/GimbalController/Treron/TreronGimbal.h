#ifndef TRERONGIMBAL_H
#define TRERONGIMBAL_H

#include <QObject>
#include "../GimbalInterface.h"
class TreronGimbal : public GimbalInterface
{
    Q_OBJECT
public:
    explicit TreronGimbal(GimbalInterface *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:
};

#endif // TRERONGIMBAL_H
