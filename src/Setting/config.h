#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <stdio.h>
#include <QObject>
#include <QVariant>
#include "../Files/FileControler.h"
#include "tinyxml2.h"
using namespace tinyxml2;
using namespace std;
class Config: public QObject
{
    Q_OBJECT
public:
    explicit Config(QObject *parent = nullptr);
    Q_INVOKABLE int readConfig(QString file);
    Q_INVOKABLE int saveConfig(QString file);
    Q_INVOKABLE int changeData(QString data,QString value);
    Q_INVOKABLE void print();
    Q_INVOKABLE QVariant getData();
    Q_INVOKABLE QVariant value(QString pattern);
    void createFile(QString fileName);
protected:
    QString m_fileConfig;
    QVariantMap m_mapData;
    XMLElement* m_settings = nullptr;
    QVariant m_data;
    XMLDocument m_doc;
};

#endif // CONFIG_H
