#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <stdio.h>
#include <QObject>
#include <QVariant>
#include "tinyxml2.h"
using namespace tinyxml2;
using namespace std;
class Config: public QObject
{
    Q_OBJECT
public:
    explicit Config(QObject *parent = nullptr);
    virtual Q_INVOKABLE int readConfig(QString file);
    virtual Q_INVOKABLE int saveConfig(QString file);
    virtual Q_INVOKABLE int changeData(QString data,QString value);
    virtual Q_INVOKABLE void print();
    virtual Q_INVOKABLE QVariant getData();
    virtual Q_INVOKABLE QVariant value(QString pattern);
public:
    QVariantMap mapData;
    QVariant m_data;
    XMLDocument m_doc;
};

#endif // CONFIG_H
