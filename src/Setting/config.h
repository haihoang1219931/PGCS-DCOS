#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <stdio.h>
#include <QObject>
#include <QVariant>
#include <QQmlListProperty>
#include "../Files/FileControler.h"
#include "tinyxml2.h"
using namespace tinyxml2;
using namespace std;

class ConfigElement : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString  name   READ name  WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QString  value       READ value      WRITE setValue NOTIFY valueChanged)
public:
    explicit ConfigElement(QObject *parent = nullptr){}
    ConfigElement(QString  name, QString  value){
        setName(name);
        setValue(value);
    }
public:
    QString name(){ return m_name;}
    QString value(){ return m_value;}
    void setName(QString value){
        if(m_name!=value){
            m_name = value;
            Q_EMIT nameChanged();
        }
    }
    void setValue(QString value){
        if(m_value!=value){
            m_value = value;
            Q_EMIT valueChanged();
        }
    }
Q_SIGNALS:
    void nameChanged();
    void valueChanged();
private:
    QString m_name;
    QString m_value;
};


class Config: public QObject
{
    Q_OBJECT
    Q_PROPERTY( QQmlListProperty<ConfigElement> paramsModel  READ paramsModel NOTIFY paramsModelChanged)
public:
    explicit Config(QObject *parent = nullptr);
public:
    QQmlListProperty<ConfigElement> paramsModel()
    {
        return QQmlListProperty<ConfigElement>(this, m_paramsModel);
    }
    Q_INVOKABLE int readConfig(QString file);
    Q_INVOKABLE int saveConfig(QString file);
    Q_INVOKABLE int changeData(QString data,QString value);
    Q_INVOKABLE void print();
    Q_INVOKABLE QVariant getData();
    Q_INVOKABLE QVariant value(QString pattern);
    Q_INVOKABLE void setPropertyValue(QString name,QString value);

Q_SIGNALS:
    void paramsModelChanged();
private:
    void createFile(QString fileName);
private:
    QList<ConfigElement*> m_paramsModel;
    QString m_fileConfig;
    QVariantMap m_mapData;
    XMLElement* m_settings = nullptr;
    QVariant m_data;
    XMLDocument m_doc;
};

#endif // CONFIG_H
