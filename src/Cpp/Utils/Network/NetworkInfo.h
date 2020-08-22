#ifndef NETWORKINFO_H
#define NETWORKINFO_H

#include <QObject>
#include <QList>
#include <QVariantMap>
#include <QNetworkConfiguration>
#include <QPair>
typedef QMap<QString, QMap<QString, QVariant>> ConnectionSetting;
Q_DECLARE_METATYPE(ConnectionSetting)
Q_DECLARE_METATYPE(QList<uint>);
Q_DECLARE_METATYPE(QList<QList<uint>>);
Q_DECLARE_METATYPE(QList<QByteArray>);
//Q_DECLARE_METATYPE(IPv6Address);
class NetworkInfo : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString  bearerTypeName READ bearerTypeName WRITE setBearerTypeName NOTIFY bearerTypeNameChanged)
    Q_PROPERTY(QString  name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(bool     activated READ activated WRITE setActivated NOTIFY activatedChanged)
    Q_PROPERTY(QString  mode READ mode WRITE setMode NOTIFY modeChanged)
    Q_PROPERTY(QString  address READ address WRITE setAddress NOTIFY addressChanged)
    Q_PROPERTY(QString  subnet READ subnet WRITE setSubnet NOTIFY subnetChanged)
    Q_PROPERTY(QString  gateway READ gateway WRITE setGateway NOTIFY gatewayChanged)
    Q_PROPERTY(QString  broadcast READ broadcast WRITE setBroadcast NOTIFY broadcastChanged)
    Q_PROPERTY(QString  setting READ setting WRITE setSetting NOTIFY settingChanged)
    Q_PROPERTY(QString  device READ device WRITE setDevice NOTIFY deviceChanged)
    Q_PROPERTY(QString  accessPoint READ accessPoint WRITE setAccessPoint NOTIFY accessPointChanged)
    Q_PROPERTY(int      frequency READ frequency WRITE setFrequency NOTIFY frequencyChanged)
    Q_PROPERTY(int      strength READ strength WRITE setStrength NOTIFY strengthChanged)
    Q_PROPERTY(bool     hasPass READ hasPass WRITE setHasPass NOTIFY hasPassChanged)
    Q_PROPERTY(ConnectionSetting    settingMap READ settingMap WRITE setSettingMap NOTIFY settingMapChanged)
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
    QString uuid(){ return m_uuid;}
    void setUUID(QString uuid){
        m_uuid = uuid;
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
    QString accessPoint(){
        return m_accessPoint;
    }
    void setAccessPoint(QString accessPoint){
        m_accessPoint = accessPoint;
        Q_EMIT accessPointChanged();
    }
    int frequency(){
        return m_frequency;
    }
    void setFrequency(int frequency){
        if(m_frequency != frequency){
            m_frequency = frequency;
            Q_EMIT frequencyChanged();
        }
    }
    int strength(){
        return m_strength;
    }
    void setStrength(int strength){
        if(m_strength != strength){
            m_strength = strength;
            Q_EMIT strengthChanged();
        }
    }
    bool hasPass(){
        return m_hasPass;
    }
    void setHasPass(bool hasPass){
        if(m_hasPass != hasPass){
            m_hasPass = hasPass;
            Q_EMIT hasPassChanged();
        }
    }
    ConnectionSetting settingMap(){return m_settingMap;}
    void setSettingMap(ConnectionSetting configMap){
        m_settingMap = configMap;
        Q_EMIT settingMapChanged();
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
    void accessPointChanged();
    void frequencyChanged();
    void strengthChanged();
    void hasPassChanged();
    void settingMapChanged();
public Q_SLOTS:

private:
    QNetworkConfiguration m_config;
    QString m_uuid;
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
    QString m_accessPoint;
    ConnectionSetting m_settingMap;
    int m_frequency = 0;
    int m_strength = 0;
    bool m_hasPass = false;
};

#endif // NETWORKINFO_H
