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
    Q_PROPERTY(QString address READ address WRITE setAddress NOTIFY addressChanged)
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
        if(m_name != name){
            m_name = name;
            Q_EMIT nameChanged();
        }
    }
    bool activated(){ return m_active;}
    void setActivated(bool active){
        if(m_active != active){
            m_active = active;
            Q_EMIT activatedChanged();
        }
    }
    QString address(){ return m_address;}
    void setAddress(QString address){
        if(m_address != address){
            m_address = address;
            Q_EMIT addressChanged();
        }
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
    }

Q_SIGNALS:
    void bearerTypeNameChanged();
    void nameChanged();
    void activatedChanged();
    void addressChanged();
public Q_SLOTS:

private:
    QNetworkConfiguration m_config;
    QString m_name;
    QString m_bearerTypeName;
    bool m_active = false;
    QString m_address;
};

#endif // NETWORKINFO_H
