#ifndef LOGCONTROLLER_H
#define LOGCONTROLLER_H

#include <QObject>
#include <QFile>
#include <iostream>
#include <fstream>

class LogController : public QObject
{
    Q_OBJECT
public:
    explicit LogController(QObject *parent = nullptr);

Q_SIGNALS:

public Q_SLOTS:
public:
    static void writeBinaryLog(QString filePath, QByteArray bytes);
};

#endif // LOGCONTROLLER_H
