#ifndef GIMBALCONTROLLER_H
#define GIMBALCONTROLLER_H

#include <QObject>
#include "GimbalData.h"
#include "GimbalInterface.h"
#include "GimbalInterfaceManager.h"
#include "Utils/Setting/config.h"
class GimbalController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(GimbalInterface* gimbal      READ gimbal             NOTIFY gimbalChanged)
    Q_PROPERTY(GimbalData* data             READ data               NOTIFY dataChanged)
public:
    explicit GimbalController(QObject *parent = nullptr);
    // property funtions
    GimbalInterface* gimbal();
    GimbalData* data();
    // other functions
    void connectToGimbal(Config* config);
    void disConnectToGimbal(Config* config);
Q_SIGNALS:
    void gimbalChanged();
    void dataChanged();
public Q_SLOTS:

public:
    GimbalInterfaceManager* m_gimbalManager = nullptr;
    GimbalInterface* m_gimbal = nullptr;
    GimbalData* m_data = nullptr;
};

#endif // GIMBALCONTROLLER_H
