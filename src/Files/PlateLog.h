#ifndef PLATELOG_H
#define PLATELOG_H

#include <QObject>
#include <QThread>
#include "PlateLogThread.h"
class PlateLog : public QObject
{
    Q_OBJECT
public:
    explicit PlateLog(QObject *parent = nullptr);

Q_SIGNALS:
    void plateReaded(QString logLine);
public Q_SLOTS:
    void appendLogFile(QString file,QString line);
    void readLogFile(QString file);
    void pause(bool pause);
private:
    QThread* m_threadReadLog;
    PlateLogThread* m_readLog;
};

#endif // PLATELOG_H
