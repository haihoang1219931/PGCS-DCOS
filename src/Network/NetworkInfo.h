#ifndef NETWORKINFO_H
#define NETWORKINFO_H

#include <QObject>
#include <QNetworkConfiguration>
class NetworkInfo : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString bearerTypeName READ bearerTypeName WRITE setBearerTypeName NOTIFY bearerTypeNameChanged)
    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(bool activated READ activated WRITE setActivated NOTIFY activatedChanged)
    Q_PROPERTY(QString mode READ mode WRITE setMode NOTIFY modeChanged)
    Q_PROPERTY(QString address READ address WRITE setAddress NOTIFY addressChanged)
    Q_PROPERTY(QString subnet READ subnet WRITE setSubnet NOTIFY subnetChanged)
    Q_PROPERTY(QString gateway READ gateway WRITE setGateway NOTIFY gatewayChanged)
    Q_PROPERTY(QString broadcast READ broadcast WRITE setBroadcast NOTIFY broadcastChanged)
    Q_PROPERTY(QString setting READ setting WRITE setSetting NOTIFY settingChanged)
    Q_PROPERTY(QString device READ device WRITE setDevice NOTIFY deviceChanged)
public:
    explicit NetworkInfo(QObject *parent = nullptr);
    QString bearerTypeName(){ return m_bearerTypeName; }
    void setBearerTypeName(QString bearerTypeName){
        if(m_bearerTypeName != bearerTypeName){
            m_bearerTypeName = bearerTypeName;
            Q_EMIT bearerTypeNameChanged();
        }
    }
    QString name(){ return m_name;}
    void setName(QString name){
        m_name = name;
        Q_EMIT nameChanged();
    }
    bool activated(){
        return m_active;
    }
    void setActivated(bool active){
        if(m_active != active){
            m_active = active;
            Q_EMIT activatedChanged();
        }
    }
    QString address(){ return m_address;}
    void setAddress(QString address){
        m_address = address;
        Q_EMIT addressChanged();
    }
    QString mode(){ return m_mode;}
    void setMode(QString mode){
        m_mode = mode;
        Q_EMIT modeChanged();
    }
    QString gateway(){ return m_gateway;}
    void setGateway(QString gateway){
        m_gateway = gateway;
        Q_EMIT gatewayChanged();
    }
    QString subnet(){ return m_subnet;}
    void setSubnet(QString subnet){
        m_subnet = subnet;
        Q_EMIT subnetChanged();
    }
    QString broadcast(){ return m_broadcast;}
    void setBroadcast(QString broadcast){
        m_broadcast = broadcast;
        Q_EMIT broadcastChanged();
    }
    QString setting(){ return m_setting;}
    void setSetting(QString setting){
        m_setting = setting;
        Q_EMIT settingChanged();
    }
    QString device(){
        return m_device;
    }
    void setDevice(QString device){
        m_device = device;
        Q_EMIT deviceChanged();
    }
    QNetworkConfiguration config(){return m_config;}
    void setConfig(QNetworkConfiguration config){
//        printf("Network[%02d] [%s] [%s] [%s] [%s]\r\n",0,
//               config.bearerTypeName().toStdString().c_str(),
//               config.name().toStdString().c_str(),
//               config.identifier().toStdString().c_str(),
//               config.state() == QNetworkConfiguration::Active? "Active":"Not active");
        m_config = config;
        setName(m_config.name());
        setActivated(m_config.state() == QNetworkConfiguration::Active);
        setBearerTypeName(m_config.bearerTypeName());
        setSetting(config.identifier());
    }

Q_SIGNALS:
    void bearerTypeNameChanged();
    void nameChanged();
    void activatedChanged();
    void addressChanged();
    void modeChanged();
    void subnetChanged();
    void gatewayChanged();
    void broadcastChanged();
    void settingChanged();
    void deviceChanged();
public Q_SLOTS:

private:
    QNetworkConfiguration m_config;
    QString m_name;
    QString m_bearerTypeName;
    bool m_active = false;
    QString m_mode;
    QString m_address;
    QString m_gateway;
    QString m_subnet;
    QString m_broadcast;
    QString m_setting;
    QString m_device;
};

#endif // NETWORKINFO_H
